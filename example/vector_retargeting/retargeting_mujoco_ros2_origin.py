import multiprocessing
import time
from pathlib import Path
from queue import Empty
from typing import Optional, Dict

import cv2
import numpy as np
import tyro
from loguru import logger

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PoseArray

import mujoco
from mujoco import viewer

from dex_retargeting.constants import (
    RobotName,
    RetargetingType,
    HandType,
    get_default_config_path,
)
from dex_retargeting.retargeting_config import RetargetingConfig
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PoseArray, Pose
from sensor_msgs.msg import JointState

OPERATOR2MANO_RIGHT = np.array(
    [
        [0, 0, -1],
        [-1, 0, 0],
        [0, 1, 0],
    ]
)

class ROS2LandmarkSubscriber(Node):
    """ROS2 节点，订阅手部关键点 PoseArray 消息，把 21x3 点塞进 queue。"""

    def __init__(self, queue: multiprocessing.Queue, topic_name: str):
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

    def landmark_callback(self, msg: PoseArray):
        try:
            # 1. 解析 21 个关键点（单位 m）
            joint_pos = np.zeros((21, 3), dtype=np.float32)
            for i, pose in enumerate(msg.poses[:21]):
                joint_pos[i] = [
                    pose.position.x / 1000.0,
                    pose.position.y / 1000.0,
                    pose.position.z / 1000.0,
                ]

            # 2. 平移到 wrist 为原点
            joint_pos = joint_pos - joint_pos[0:1, :]

            # 3. 用 3 个点估计手掌坐标系旋转
            wrist_rot = self.estimate_frame_from_hand_points(joint_pos)
            # 4. 旋转 + MANO 坐标变换，仍然是 21x3
            joint_pos = joint_pos @ wrist_rot @ self.operator2mano

            # 5. 丢进队列
            if self.queue.full():
                try:
                    self.queue.get_nowait()
                except Empty:
                    pass
            self.queue.put_nowait(joint_pos)

        except Exception as e:
            logger.error(f"Error processing PoseArray: {e}")

    @staticmethod
    def estimate_frame_from_hand_points(keypoint_3d_array: np.ndarray) -> np.ndarray:
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



def build_mujoco_joint_mapping(model: mujoco.MjModel) -> Dict[str, int]:
    """
    建立 MuJoCo 里 joint name -> qpos index 的映射。
    只管 1-DoF hinge 关节，free joint 之类的跳过。
    """
    name_to_qpos = {}
    for j in range(model.njnt):
        jtype = model.jnt_type[j]
        # 只要 hinge 关节（手指基本都是 hinge）
        if jtype != mujoco.mjtJoint.mjJNT_HINGE:
            continue
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, j)
        if name is None:
            continue
        qpos_adr = model.jnt_qposadr[j]
        name_to_qpos[name] = qpos_adr
    logger.info(f"MuJoCo hinge joints: {list(name_to_qpos.keys())}")
    return name_to_qpos


def start_retargeting_mujoco(
    queue: multiprocessing.Queue,
    robot_dir: str,
    config_path: str,
    mjcf_path: str,
):
    """
    消费 queue 里的 21x3 关键点：
    -> retargeting.retarget(ref_value) 得到 robot qpos
    -> 写到 MuJoCo 的 data.qpos 里
    -> MuJoCo viewer 动画。
    """
    # --- init retargeting ---
    rclpy.init()
    RetargetingConfig.set_default_urdf_dir(str(robot_dir))
    logger.info(f"Start retargeting with config {config_path}")
    retargeting_cfg = RetargetingConfig.load_from_file(config_path)
    retargeting = retargeting_cfg.build()

    # retargeting 侧的 DOF 名称
    dof_names = list(retargeting.optimizer.robot.dof_joint_names)
    logger.info(f"Retargeting DOF joints: {dof_names}")

    # --- init MuJoCo ---
    mj_model = mujoco.MjModel.from_xml_path(mjcf_path)
    mj_model.opt.gravity[:] = 0
    mj_data = mujoco.MjData(mj_model)
    name_to_qpos = build_mujoco_joint_mapping(mj_model)


    # 打个日志看名字对没对齐
    missing = [n for n in dof_names if n not in name_to_qpos]
    if missing:
        logger.warning(
            f"The following retargeting DOF joints not found in MuJoCo model: {missing}"
        )

    # --- 主循环：MuJoCo viewer + retargeting ---
    try:
        with viewer.launch_passive(mj_model, mj_data) as v:
            logger.info("MuJoCo viewer launched.")
            target_dt = 1.0 / 60.0
            last = time.time()
            while v.is_running():
                now = time.time()
                dt = now - last
                if dt < target_dt:
                    time.sleep(target_dt - dt)
                    now = time.time()
                latest = None
                try:
                    # 把队列里所有旧帧都清掉，只留最后一帧
                    while True:
                        latest = queue.get_nowait()
                except Empty:
                    pass

                if latest is not None:
                    joint_pos = latest
                else:
                    joint_pos = None

                if joint_pos is not None:
                    # 和你原来代码一致的 ref_value 逻辑
                    retargeting_type = retargeting.optimizer.retargeting_type
                    indices = retargeting.optimizer.target_link_human_indices

                    if retargeting_type == "POSITION":
                        ref_value = joint_pos[indices, :]
                    else:
                        origin_indices = indices[0, :]
                        task_indices = indices[1, :]
                        ref_value = (
                            joint_pos[task_indices, :] - joint_pos[origin_indices, :]
                        )

                    qpos = retargeting.retarget(ref_value).astype(np.float64)


                    # 一些简单的 joint 限幅（示例，和你原来类似）
                    def clip_joint(
                        qpos_arr,
                        joint_names,
                        joint_name,
                        lo,
                        hi,
                        scale=1.0,
                        bias=0.0,
                    ):
                        if joint_name in joint_names:
                            idx = joint_names.index(joint_name)
                            val = qpos_arr[idx] * scale + bias
                            qpos_arr[idx] = float(np.clip(val, lo, hi))

                    clip_joint(
                        qpos,
                        dof_names,
                        joint_name="R_pinky_pip_joint",
                        lo=-1.0,
                        hi=1.6,
                        scale=0.8,
                    )
                    clip_joint(
                        qpos,
                        dof_names,
                        joint_name="R_thumb_roll_joint",
                        lo=-0.2,
                        hi=0.8,
                        scale=1.2,
                    )
                    clip_joint(
                        qpos,
                        dof_names,
                        joint_name="R_thumb_mcp_joint",
                        lo=-0.2,
                        hi=0.8,
                        scale=1.2,
                    )

                    # 写回 MuJoCo 的 qpos
                    for i, name in enumerate(dof_names):
                        if name not in name_to_qpos:
                            continue
                        adr = name_to_qpos[name]
                        mj_data.qpos[adr] = qpos[i]
                else:
                    pass

                n_substeps = 5
                for _ in range(n_substeps):
                    mujoco.mj_step(mj_model, mj_data)

                # 同步到 viewer
                v.sync()

    except KeyboardInterrupt:
        logger.info("Retargeting-MuJoCo loop interrupted by user.")
    finally:
        rclpy.shutdown()
        logger.info("ROS2 shutdown.")


def produce_frame(queue: multiprocessing.Queue, ros_topic: Optional[str] = None):
    """ROS2 订阅进程：只负责往 queue 塞 21x3 关键点。"""
    if ros_topic is None:
        ros_topic = "/vrpn/hand_kp"

    rclpy.init()
    node = ROS2LandmarkSubscriber(queue, ros_topic)
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()
        logger.info("ROS2 landmark subscriber shutdown.")


def main(
    robot_name: RobotName,
    retargeting_type: RetargetingType,
    hand_type: HandType,
    mjcf_path: str,
    ros_topic: str = "/vrpn/hand_kp",
):
    """
    基于原来的 retargeting_ros2.py，
    只是把关节输出从 ROS joint_commands 换成 MuJoCo 模型的 qpos。
    """
    config_path = get_default_config_path(robot_name, retargeting_type, hand_type)
    robot_dir = (
        Path(__file__).absolute().parent.parent.parent / "assets" / "robots" / "hands"
    )

    queue = multiprocessing.Queue(maxsize=10)

    producer_process = multiprocessing.Process(
        target=produce_frame,
        args=(queue, ros_topic),
    )
    consumer_process = multiprocessing.Process(
        target=start_retargeting_mujoco,
        args=(queue, str(robot_dir), str(config_path), mjcf_path),
    )

    producer_process.start()
    logger.info("Started ROS2 landmark producer process.")

    time.sleep(2)  # 给 ROS2 节点一点启动时间

    consumer_process.start()
    logger.info("Started retargeting + MuJoCo consumer process.")

    producer_process.join()
    consumer_process.join()

    logger.info("All processes finished.")


if __name__ == "__main__":
    tyro.cli(main)
