import multiprocessing
import time
from pathlib import Path
from queue import Empty
from typing import Optional
import csv

import cv2
import numpy as np
import tyro
from loguru import logger

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PoseArray
from sensor_msgs.msg import JointState

from dex_retargeting.constants import (
    RobotName,
    RetargetingType,
    HandType,
    get_default_config_path,
)
from dex_retargeting.retargeting_config import RetargetingConfig

OPERATOR2MANO_RIGHT = np.array(
    [
        [0, 0, -1],
        [-1, 0, 0],
        [0, 1, 0],
    ]
)


# ================= ROS2 节点：发布关节命令 =================
class JointStatePublisher(Node):
    def __init__(self, robot_name: str, joint_log_path: Path):
        super().__init__("retargeting_joint_publisher")
        self.publisher_ = self.create_publisher(JointState, "/joint_commands", 10)
        self.joint_names = []  # 将在后续填充
        self.robot_name = robot_name
        self.get_logger().info(f"JointState publisher created for {robot_name}")

        # 日志文件：joint_commands_log.csv
        joint_log_path.parent.mkdir(parents=True, exist_ok=True)
        self.joint_log_file = joint_log_path.open("w", newline="")
        self.joint_log_writer = csv.writer(self.joint_log_file)
        self.joint_log_writer.writerow(
            ["stamp_sec", "stamp_nanosec", "frame_id", "joint_name", "position"]
        )
        self.frame_id = 0

    def publish_joints(self, positions: np.ndarray):
        # 1) 组 JointState 消息
        stamp = self.get_clock().now().to_msg()

        msg = JointState()
        msg.header.stamp = stamp
        msg.header.frame_id = self.robot_name + "_base"
        msg.name = list(self.joint_names)
        msg.position = positions.tolist()
        self.publisher_.publish(msg)

        # 2) 同时写日志：每一行一个关节
        for name, pos in zip(self.joint_names, positions):
            self.joint_log_writer.writerow(
                [stamp.sec, stamp.nanosec, self.frame_id, name, float(pos)]
            )
        self.joint_log_file.flush()
        self.frame_id += 1

    def destroy_node(self):
        try:
            self.joint_log_file.close()
        except Exception:
            pass
        super().destroy_node()


# ================= ROS2 节点：订阅 21 点 landmarks，并记录原始坐标 =================
class ROS2LandmarkSubscriber(Node):
    """订阅手部 PoseArray：一方面做坐标变换塞到 queue，另一方面记录原始 21x3 点到 CSV。"""

    def __init__(self, queue: multiprocessing.Queue, topic_name: str, landmark_log_path: Path):
        super().__init__("hand_retargeting_node")
        self.queue = queue
        self.operator2mano = OPERATOR2MANO_RIGHT

        self.subscription = self.create_subscription(
            PoseArray,
            topic_name,
            self.landmark_callback,
            10,
        )
        logger.info(f"Subscribed to ROS2 topic: {topic_name}")

        # 日志文件：hotrack_landmarks_log.csv
        landmark_log_path.parent.mkdir(parents=True, exist_ok=True)
        self.landmark_log_file = landmark_log_path.open("w", newline="")
        self.landmark_log_writer = csv.writer(self.landmark_log_file)
        # header: stamp + 21*3
        header = ["stamp_sec", "stamp_nanosec"]
        for i in range(21):
            header += [f"p{i}_x", f"p{i}_y", f"p{i}_z"]
        self.landmark_log_writer.writerow(header)

    def landmark_callback(self, msg: PoseArray):
        """处理 PoseArray 消息：记录原始坐标 + 推 MANO 坐标进队列。"""
        try:
            # --- 1) 记录原始 21 点（单位 m，未做 wrist 对齐 / 旋转）到 CSV ---
            joint_pos_sensor = np.zeros((21, 3), dtype=np.float32)
            for i, pose in enumerate(msg.poses[:21]):
                joint_pos_sensor[i] = [
                    pose.position.x / 1000.0,
                    pose.position.y / 1000.0,
                    pose.position.z / 1000.0,
                ]

            row = [msg.header.stamp.sec, msg.header.stamp.nanosec]
            for i in range(21):
                row += joint_pos_sensor[i].tolist()
            self.landmark_log_writer.writerow(row)
            self.landmark_log_file.flush()

            # --- 2) 做和以前一样的坐标处理，塞 MANO 坐标进 queue ---
            joint_pos = joint_pos_sensor.copy()
            # 以 wrist 为原点
            joint_pos = joint_pos - joint_pos[0:1, :]
            wrist_rot = self.estimate_frame_from_hand_points(joint_pos)
            joint_pos = joint_pos @ wrist_rot @ self.operator2mano  # MANO 坐标系

            # 丢掉旧数据，只保留最新
            if self.queue.full():
                try:
                    self.queue.get_nowait()
                except Empty:
                    pass
            self.queue.put_nowait(joint_pos)

        except Exception as e:
            logger.error(f"Error processing PoseArray: {str(e)}")

    @staticmethod
    def estimate_frame_from_hand_points(keypoint_3d_array: np.ndarray) -> np.ndarray:
        """
        从 21 点估计手掌坐标系（和你原来的版本一致）。
        """
        assert keypoint_3d_array.shape == (21, 3)
        points = keypoint_3d_array[[0, 5, 9], :]

        x_vector = points[0] - points[2]

        points = points - np.mean(points, axis=0, keepdims=True)
        _, _, v = np.linalg.svd(points)
        normal = v[2, :]

        x = x_vector - np.sum(x_vector * normal) * normal
        x = x / np.linalg.norm(x)
        z = np.cross(x, normal)

        if np.sum(z * (points[1] - points[2])) < 0:
            normal *= -1
            z *= -1
        frame = np.stack([x, normal, z], axis=1)
        return frame

    def destroy_node(self):
        try:
            self.landmark_log_file.close()
        except Exception:
            pass
        super().destroy_node()


# ================= retargeting + 发布关节 + 记录优化器内部数据 =================
def start_retargeting(queue: multiprocessing.Queue, robot_dir: str, config_path: str, log_dir: str):
    rclpy.init()
    RetargetingConfig.set_default_urdf_dir(str(robot_dir))
    logger.info(f"Start retargeting with config {config_path}")

    # 构建 retargeting
    retargeting_cfg = RetargetingConfig.load_from_file(config_path)
    retargeting = retargeting_cfg.build()
    config = retargeting_cfg  # 这里直接复用，不再重新 load 一次

    # 从 URDF 路径拿一个 robot_name（不用再通过 sapien）
    urdf_path = Path(config.urdf_path)
    robot_name = urdf_path.stem

    # 创建 joint publisher + 日志
    log_dir_path = Path(log_dir)
    joint_log_path = log_dir_path / "joint_commands_log.csv"
    opt_log_path = log_dir_path / "retarget_internal_log.csv"

    joint_publisher_node = JointStatePublisher(robot_name, joint_log_path)

    # retargeting 里的关节名字
    retargeting_joint_names = retargeting.joint_names               # full DOF
    actuated_joint_names = retargeting.optimizer.target_joint_names # 会真的发给真机的 DOF

    # 找出 actuated joints 在 full qpos 里的 index
    idx_publish = np.array(
        [retargeting_joint_names.index(name) for name in actuated_joint_names],
        dtype=int,
    )

    # JointState 里就直接用 actuated_joint_names
    joint_publisher_node.joint_names = list(actuated_joint_names)

    # 找小拇指 pip 关节在 actuated_joint_names 里的 index（如果存在）
    try:
        pinky_joint_name = "R_pinky_pip_joint"
        pinky_idx_publish = actuated_joint_names.index(pinky_joint_name)
        logger.info(f"{pinky_joint_name} index in published joints = {pinky_idx_publish}")
    except ValueError:
        pinky_joint_name = None
        pinky_idx_publish = None
        logger.warning("R_pinky_pip_joint not found in actuated_joint_names!")

    # retarget internal log（按帧记录 ref_value + 全量 qpos + pinky 角）
    opt_log_path.parent.mkdir(parents=True, exist_ok=True)
    opt_log_file = opt_log_path.open("w", newline="")
    opt_writer = csv.writer(opt_log_file)

    header = ["frame_id", "time_sec", "time_ns"]
    # ref_value 平铺
    header += [f"ref_{i}" for i in range(retargeting.optimizer.target_link_human_indices.size * 3)]
    # full qpos
    header += [f"qpos_{name}" for name in retargeting_joint_names]
    # published joints
    header += [f"qpub_{name}" for name in actuated_joint_names]
    # pinky pip angle（如果有）
    header += ["pinky_pip_angle"]
    opt_writer.writerow(header)

    frame_id = 0

    try:
        while rclpy.ok():
            try:
                # 从队列获取 MANO 坐标下的 21x3 手势点
                joint_pos = queue.get(timeout=1.0)
            except Empty:
                logger.warning("No hand data received in 1 second. Applying default pose.")
                rclpy.spin_once(joint_publisher_node, timeout_sec=0.001)
                continue

            # === 1) 计算 ref_value（DexPilot 的 target_vector） ===
            opt = retargeting.optimizer
            retargeting_type = opt.retargeting_type
            indices = opt.target_link_human_indices

            if retargeting_type == "POSITION":
                ref_value = joint_pos[indices, :]  # (K, 3)
            else:
                origin_indices = indices[0, :]
                task_indices = indices[1, :]
                ref_value = joint_pos[task_indices, :] - joint_pos[origin_indices, :]

            # 展平 ref_value 方便存 CSV
            ref_flat = ref_value.reshape(-1)

            # === 2) 调用 retargeting 得到 qpos（full DOF） ===
            qpos = retargeting.retarget(ref_value)  # full DOF, 对应 retargeting_joint_names
            qpos = np.asarray(qpos, dtype=np.float32)

            # 取 actuated joints 顺序
            qpos_publish = qpos[idx_publish]

            # === 3) 发布到 /joint_commands ===
            joint_publisher_node.publish_joints(qpos_publish)

            # 当前时间（非 ROS 时间，用于内部分析）
            now = time.time()
            now_sec = int(now)
            now_ns = int((now - now_sec) * 1e9)

            # === 4) 记录 retarget 内部数据到 CSV ===
            row = [frame_id, now_sec, now_ns]
            row += ref_flat.tolist()
            row += qpos.tolist()
            row += qpos_publish.tolist()
            # pinky pip 单独一列（如果存在）
            if pinky_idx_publish is not None:
                pinky_angle = float(qpos_publish[pinky_idx_publish])
            else:
                pinky_angle = float("nan")
            row.append(pinky_angle)
            opt_writer.writerow(row)
            opt_log_file.flush()

            frame_id += 1

            # 让 ROS2 处理 callback
            rclpy.spin_once(joint_publisher_node, timeout_sec=0.001)

    except KeyboardInterrupt:
        logger.info("Retargeting loop interrupted by user.")
    finally:
        opt_log_file.close()
        joint_publisher_node.destroy_node()
        rclpy.shutdown()


def produce_frame(queue: multiprocessing.Queue, ros_topic: Optional[str], log_dir: str):
    """
    订阅 /hotrack/landmarks（或你指定的 topic），
    一边把 MANO 坐标系下 21x3 丢进 queue，
    一边把“原始传感器 21 点”写到 CSV。
    """
    if ros_topic is None:
        ros_topic = "/hotrack/landmarks"

    rclpy.init()

    log_dir_path = Path(log_dir)
    landmark_log_path = log_dir_path / "hotrack_landmarks_log.csv"

    pos_subscriber = ROS2LandmarkSubscriber(queue, ros_topic, landmark_log_path)

    try:
        rclpy.spin(pos_subscriber)
    except KeyboardInterrupt:
        pass
    finally:
        pos_subscriber.destroy_node()
        rclpy.shutdown()


def main(
    robot_name: RobotName,
    retargeting_type: RetargetingType,
    hand_type: HandType,
    ros_topic: str = "/hotrack/landmarks",
    log_dir: str = "data",
):
    """
    在线 retarget + 真机控制 + 全链路数据记录。

    会在 log_dir 下生成：
      - hotrack_landmarks_log.csv      # 原始 21 点（传感器坐标）
      - joint_commands_log.csv         # /joint_commands 发布的关节角
      - retarget_internal_log.csv      # ref_value + full qpos + pinky_pip_angle
    """
    config_path = get_default_config_path(robot_name, retargeting_type, hand_type)
    robot_dir = (
        Path(__file__).absolute().parent.parent.parent / "assets" / "robots" / "hands"
    )
    queue = multiprocessing.Queue(maxsize=10)

    producer_process = multiprocessing.Process(
        target=produce_frame, args=(queue, ros_topic, log_dir)
    )
    consumer_process = multiprocessing.Process(
        target=start_retargeting, args=(queue, str(robot_dir), str(config_path), log_dir)
    )

    producer_process.start()
    logger.info("Started producer process")

    # 给 ROS2 节点一点启动时间
    time.sleep(2)
    consumer_process.start()
    logger.info("Started consumer process")

    producer_process.join()
    consumer_process.join()
    time.sleep(1)

    print("done")


if __name__ == "__main__":
    tyro.cli(main)
