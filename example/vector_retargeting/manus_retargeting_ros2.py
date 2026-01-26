import multiprocessing
import time
from pathlib import Path
from queue import Empty
from collections import deque
from typing import Optional

import cv2
import numpy as np
import sapien
import tyro
from loguru import logger
from sapien.asset import create_dome_envmap
from sapien.utils import Viewer

from dex_retargeting.constants import (
    RobotName,
    RetargetingType,
    HandType,
    get_default_config_path,
)
from dex_retargeting.retargeting_config import RetargetingConfig


import rclpy
from rclpy.node import Node
from manus_ros2_msgs.msg import ManusNodePoses
from sensor_msgs.msg import JointState

OPERATOR2MANO_RIGHT = np.array(
    [
        [0, 0, -1],
        [-1, 0, 0],
        [0, 1, 0],
    ]
)

# 创建ROS2节点和发布者
class JointStatePublisher(Node):
    def __init__(self, robot_name: str, topic_name: str):
        super().__init__('retargeting_joint_publisher')
        self.publisher_ = self.create_publisher(
            JointState,
            topic_name,
            10
        )
        self.joint_names = []  # 将在后续填充
        self.robot_name = robot_name
        self.get_logger().info(
            f"JointState publisher created for {robot_name} on {topic_name}"
        )
        
    def publish_joints(self, positions):
        msg = JointState()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = self.robot_name + "_base"
        msg.name = self.joint_names
        msg.position = positions.tolist()  # 转换为Python list
        self.publisher_.publish(msg)

MANUS_REMOVED_NODE_IDS = {5, 10, 15, 20}
MANUS_EXPECTED_NODE_IDS = [
    idx for idx in range(25) if idx not in MANUS_REMOVED_NODE_IDS
]


class ROS2ManusNodeSubscriber(Node):
    """ROS2节点，订阅ManusNodePoses消息"""

    def __init__(
        self,
        queue: multiprocessing.Queue,
        topic_name: str,
        position_scale: float = 1.0,
    ):
        super().__init__("hand_retargeting_node")
        self.queue = queue
        self.operator2mano = OPERATOR2MANO_RIGHT
        self.position_scale = position_scale
        self.subscription = self.create_subscription(
            ManusNodePoses,
            topic_name,
            self.landmark_callback,
            10
        )
        logger.info(f"Subscribed to ROS2 topic: {topic_name}")

    def landmark_callback(self, msg: ManusNodePoses):
        """处理ManusNodePoses消息的回调函数"""
        try:
            if len(msg.node_ids) != len(msg.poses):
                logger.warning(
                    "ManusNodePoses node_ids/poses length mismatch: "
                    f"{len(msg.node_ids)} vs {len(msg.poses)}"
                )
                return

            pose_by_id = {nid: pose for nid, pose in zip(msg.node_ids, msg.poses)}
            missing = [nid for nid in MANUS_EXPECTED_NODE_IDS if nid not in pose_by_id]
            if missing:
                logger.warning(f"ManusNodePoses missing node_ids: {missing}")
                return

            # 解析25个点并剔除5/10/15/20 -> 21个关键点的位置
            joint_pos = np.zeros((21, 3), dtype=np.float32)
            for i, node_id in enumerate(MANUS_EXPECTED_NODE_IDS):
                pose = pose_by_id[node_id]
                joint_pos[i] = [
                    pose.position.x * self.position_scale,
                    pose.position.y * self.position_scale,
                    pose.position.z * self.position_scale,
                ]
            
            # 放入队列供重定向进程使用
            if self.queue.full():
                try:
                    self.queue.get_nowait()  # 丢弃最旧数据
                except Empty:
                    pass
            joint_pos = joint_pos-joint_pos[0:1, :]
            wrist_rot = self.estimate_frame_from_hand_points(joint_pos)
            joint_pos = joint_pos @ wrist_rot @ self.operator2mano
            self.queue.put_nowait(joint_pos)
        except Exception as e:
            logger.error(f"Error processing ManusNodePoses: {str(e)}")

    @staticmethod
    def estimate_frame_from_hand_points(keypoint_3d_array: np.ndarray) -> np.ndarray:
        """
        Compute the 3D coordinate frame (orientation only) from detected 3d key points
        :param points: keypoint3 detected from MediaPipe detector. Order: [wrist, index, middle, pinky]
        :return: the coordinate frame of wrist in MANO convention
        """
        assert keypoint_3d_array.shape == (21, 3)
        points = keypoint_3d_array[[0, 5, 9], :]

        # Compute vector from palm to the first joint of middle finger
        x_vector = points[0] - points[2]

        # Normal fitting with SVD
        points = points - np.mean(points, axis=0, keepdims=True)
        u, s, v = np.linalg.svd(points)

        normal = v[2, :]

        # Gram–Schmidt Orthonormalize
        x = x_vector - np.sum(x_vector * normal) * normal
        x = x / np.linalg.norm(x)
        z = np.cross(x, normal)

        # We assume that the vector from pinky to index is similar the z axis in MANO convention
        if np.sum(z * (points[1] - points[2])) < 0:
            normal *= -1
            z *= -1
        frame = np.stack([x, normal, z], axis=1)
        return frame

def _gaussian_weights(window_size: int, sigma: float) -> np.ndarray:
    if window_size <= 1:
        return np.ones(1)
    half = window_size // 2
    x = np.arange(-half, half + 1)
    weights = np.exp(-(x ** 2) / (2 * sigma ** 2))
    return weights / np.sum(weights)

FINGER_DIP_INDICES = np.array([3, 7, 11, 15, 19], dtype=int)
FINGER_TIP_INDICES = np.array([4, 8, 12, 16, 20], dtype=int)


def _fingertip_direction_vectors(
    joint_pos: np.ndarray, num_fingers: int
) -> np.ndarray:
    dip_idx = FINGER_DIP_INDICES[:num_fingers]
    tip_idx = FINGER_TIP_INDICES[:num_fingers]
    vec = joint_pos[tip_idx, :] - joint_pos[dip_idx, :]
    norm = np.linalg.norm(vec, axis=1, keepdims=True) + 1e-6
    return vec / norm


def _append_fingertip_direction(
    ref_value: np.ndarray, joint_pos: np.ndarray, optimizer
) -> np.ndarray:
    if getattr(optimizer, "fingertip_direction_weight", 0.0) <= 0.0:
        return ref_value
    if getattr(optimizer, "finger_dip_link_names", None) is None:
        return ref_value
    dir_vec = _fingertip_direction_vectors(joint_pos, optimizer.num_fingers)
    return np.concatenate([ref_value, dir_vec], axis=0)

def start_retargeting(
    queue: multiprocessing.Queue,
    robot_dir: str,
    config_path: str,
    publish_topic: str,
    publish_rate_hz: float,
    filter_type: str,
    filter_window: int,
    filter_sigma: float,
    ema_alpha: float,
    max_vel_rad: float,
    anti_flip_enable: bool,
    anti_flip_min_rad: float,
    anti_flip_blend: float,
    anti_flip_strict: bool,
    anti_flip_use_urdf_limits: bool,
    anti_flip_lower_offset_rad: float,
    anti_flip_hard_limit: bool,
    anti_flip_extra_joint4_fingers: str,
    anti_flip_fingers: str,
    anti_flip_seed_straight: bool,
):
    rclpy.init()
    RetargetingConfig.set_default_urdf_dir(str(robot_dir))
    logger.info(f"Start retargeting with config {config_path}")
    retargeting = RetargetingConfig.load_from_file(config_path).build()
    config = RetargetingConfig.load_from_file(config_path)

    scene = sapien.Scene()
    loader = scene.create_urdf_loader()
    filepath = Path(config.urdf_path)
    robot_name = filepath.stem
    if "glb" not in robot_name:
        filepath = str(filepath).replace(".urdf", "_glb.urdf")
    else:
        filepath = str(filepath)
    robot = loader.load(filepath)
    joint_publisher_node = JointStatePublisher(robot_name, publish_topic)

    # Different robot loader may have different orders for joints
    sapien_joint_names = [joint.get_name() for joint in robot.get_active_joints()]
    joint_publisher_node.joint_names = sapien_joint_names  # 设置发布者的关节名称
    retargeting_joint_names = retargeting.joint_names
    actuated_joint_names = retargeting.optimizer.target_joint_names  # 或从 yml 读

    idx_publish = np.array(
    [retargeting_joint_names.index(name) for name in actuated_joint_names],
    dtype=int,
    )

    finger_name_groups = {}
    if anti_flip_fingers:
        try:
            finger_ids = {
                int(x.strip())
                for x in anti_flip_fingers.split(",")
                if x.strip()
            }
        except ValueError:
            finger_ids = set()
        for finger_id in finger_ids:
            target_name = f"finger{finger_id}_joint3"
            indices = [
                i
                for i, name in enumerate(actuated_joint_names)
                if name == target_name
            ]
            if indices:
                finger_name_groups[finger_id] = indices

    if anti_flip_extra_joint4_fingers:
        try:
            finger_ids = {
                int(x.strip())
                for x in anti_flip_extra_joint4_fingers.split(",")
                if x.strip()
            }
        except ValueError:
            finger_ids = set()
        for finger_id in finger_ids:
            target_name = f"finger{finger_id}_joint4"
            indices = [
                i
                for i, name in enumerate(actuated_joint_names)
                if name == target_name
            ]
            if indices:
                finger_name_groups.setdefault(finger_id, []).extend(indices)

    joint_lower_bounds = {}
    if anti_flip_use_urdf_limits:
        for i, name in enumerate(retargeting.optimizer.target_joint_names):
            lower = retargeting.joint_limits[i, 0]
            joint_lower_bounds[name] = lower + anti_flip_lower_offset_rad

    if anti_flip_hard_limit and finger_name_groups:
        new_limits = retargeting.joint_limits.copy()
        for indices in finger_name_groups.values():
            for idx in indices:
                new_limits[idx, 0] = new_limits[idx, 0] + anti_flip_lower_offset_rad
        retargeting.joint_limits = new_limits
        retargeting.optimizer.set_joint_limit(new_limits)
        retargeting.last_qpos = np.clip(
            retargeting.last_qpos, new_limits[:, 0], new_limits[:, 1]
        )

    history = deque(maxlen=max(1, filter_window))
    gaussian_weights = None
    if filter_type == "gaussian":
        window_size = max(1, filter_window)
        if window_size % 2 == 0:
            window_size += 1
        history = deque(maxlen=window_size)
        gaussian_weights = _gaussian_weights(window_size, filter_sigma)

    last_qpos = None
    last_time = None
    if anti_flip_seed_straight:
        last_good_qpos = np.zeros(len(actuated_joint_names), dtype=np.float32)
    else:
        last_good_qpos = None

# -------------------------------------------------------------------
    prev_qpos = None
    prev_time = None
    curr_qpos = None
    curr_time = None
    target_dt = 1.0 / max(1e-3, publish_rate_hz)
    last_pub = time.time()
    while True:
        # Consume latest joint data if available.
        try:
            while True:
                joint_pos = queue.get_nowait()
                retargeting_type = retargeting.optimizer.retargeting_type
                indices = retargeting.optimizer.target_link_human_indices

                if retargeting_type == "POSITION":
                    ref_value = joint_pos[indices, :]
                else:
                    origin_indices = indices[0, :]
                    task_indices = indices[1, :]
                    ref_value = joint_pos[task_indices, :] - joint_pos[origin_indices, :]
                    if retargeting_type == "DEXPILOT":
                        ref_value = _append_fingertip_direction(
                            ref_value, joint_pos, retargeting.optimizer
                        )
                    qpos = retargeting.retarget(ref_value)  # full DOF
                    qpos_publish = qpos[idx_publish]        # 只取 10 个

                    prev_qpos, prev_time = curr_qpos, curr_time
                    curr_qpos = qpos_publish
                    curr_time = time.time()
        except Empty:
            pass

        now = time.time()
        if now - last_pub >= target_dt:
            last_pub = now

            if curr_qpos is not None:
                if (
                    prev_qpos is not None
                    and prev_time is not None
                    and curr_time is not None
                    and curr_time > prev_time
                ):
                    alpha = (now - prev_time) / (curr_time - prev_time)
                    alpha = float(np.clip(alpha, 0.0, 1.0))
                    qpos_publish = prev_qpos + alpha * (curr_qpos - prev_qpos)
                else:
                    qpos_publish = curr_qpos

                if filter_type == "ema":
                    if len(history) == 0:
                        history.append(qpos_publish)
                    else:
                        last = history[-1]
                        smoothed = ema_alpha * qpos_publish + (1 - ema_alpha) * last
                        history.append(smoothed)
                    qpos_publish = history[-1]
                elif filter_type == "gaussian":
                    history.append(qpos_publish)
                    if len(history) == history.maxlen:
                        stacked = np.stack(history, axis=0)
                        qpos_publish = np.sum(
                            stacked * gaussian_weights[:, None], axis=0
                        )

                if max_vel_rad > 0:
                    if last_time is None:
                        last_time = now
                        last_qpos = qpos_publish.copy()
                    else:
                        dt = max(1e-6, now - last_time)
                        max_delta = max_vel_rad * dt
                        delta = qpos_publish - last_qpos
                        delta = np.clip(delta, -max_delta, max_delta)
                        qpos_publish = last_qpos + delta
                        last_qpos = qpos_publish.copy()
                        last_time = now

                if anti_flip_enable and finger_name_groups:
                    if last_good_qpos is None:
                        last_good_qpos = qpos_publish.copy()
                    abnormal_indices = []
                    for finger_id, indices in finger_name_groups.items():
                        finger_vals = qpos_publish[indices]
                        for idx in indices:
                            name = actuated_joint_names[idx]
                            lower = joint_lower_bounds.get(name, anti_flip_min_rad)
                            if qpos_publish[idx] < lower:
                                abnormal_indices.append(idx)
                    if abnormal_indices:
                        qpos_corrected = qpos_publish.copy()
                        qpos_corrected[abnormal_indices] = (
                            (1.0 - anti_flip_blend) * qpos_corrected[abnormal_indices]
                            + anti_flip_blend * last_good_qpos[abnormal_indices]
                        )
                        if anti_flip_strict:
                            for idx in abnormal_indices:
                                name = actuated_joint_names[idx]
                                lower = joint_lower_bounds.get(name, anti_flip_min_rad)
                                qpos_corrected[idx] = max(
                                    qpos_corrected[idx], lower
                                )
                        qpos_publish = qpos_corrected
                        retargeting.last_qpos = qpos_publish.astype(np.float32)
                    else:
                        last_good_qpos = qpos_publish.copy()

                joint_publisher_node.joint_names = list(actuated_joint_names)
                joint_publisher_node.publish_joints(qpos_publish)
        # 检查退出条件
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
        rclpy.spin_once(joint_publisher_node, timeout_sec=0.001)
    # 清理
    joint_publisher_node.destroy_node()
    rclpy.shutdown()

def produce_frame(
    queue: multiprocessing.Queue,
    ros_topic: Optional[str] = None,
    position_scale: float = 1.0,
):
    if ros_topic is None:
        ros_topic = "/manus_node_poses_0"  # 默认话题
    
    rclpy.init()
    pos_subscriber = ROS2ManusNodeSubscriber(
        queue, ros_topic, position_scale=position_scale
    )
    
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
    ros_topic: str = "/manus_node_poses_0",  # 实际topic
    publish_topic: str = "/joint_commands",
    publish_rate_hz: float = 60.0,
    filter_type: str = "ema",
    filter_window: int = 7,
    filter_sigma: float = 1.5,
    ema_alpha: float = 0.1,
    max_vel_rad: float = 1.0,
    anti_flip_enable: bool = True,
    anti_flip_min_rad: float = -0.2,
    anti_flip_blend: float = 0.7,
    anti_flip_strict: bool = True,
    anti_flip_use_urdf_limits: bool = True,
    anti_flip_lower_offset_rad: float = 0.3,
    anti_flip_hard_limit: bool = True,
    anti_flip_extra_joint4_fingers: str = "1,5",
    anti_flip_fingers: str = "1,3,4,5",
    anti_flip_seed_straight: bool = True,
    position_scale: float = 1.0,
):
    """
    Detects the human hand pose from a video and translates the human pose trajectory into a robot pose trajectory.

    Args:
        robot_name: The identifier for the robot. This should match one of the default supported robots.
        retargeting_type: The type of retargeting, each type corresponds to a different retargeting algorithm.
        hand_type: Specifies which hand is being tracked, either left or right.
            Please note that retargeting is specific to the same type of hand: a left robot hand can only be retargeted
            to another left robot hand, and the same applies for the right hand.
        ros_topic: the topic name to get joints info
        publish_topic: the topic name to publish JointState
        publish_rate_hz: publish rate for command output
        filter_type: "none", "ema", or "gaussian"
        filter_window: window size for gaussian filter (odd number recommended)
        filter_sigma: sigma for gaussian filter
        ema_alpha: smoothing factor for EMA, larger is less smooth
        max_vel_rad: max joint velocity (rad/s), <= 0 disables the limiter
        anti_flip_enable: enable anti-hyperextension correction
        anti_flip_min_rad: threshold for detecting reverse bend (rad)
        anti_flip_blend: blend ratio to last good pose for bad joints
        anti_flip_strict: clamp bad joints to anti_flip_min_rad after blending
        anti_flip_use_urdf_limits: use URDF lower limits as threshold
        anti_flip_lower_offset_rad: raise URDF lower limits by this offset
        anti_flip_hard_limit: enforce raised lower limits in the optimizer
        anti_flip_extra_joint4_fingers: comma-separated finger ids to also guard joint4
        anti_flip_fingers: comma-separated finger ids to guard (1-5)
        anti_flip_seed_straight: initialize last good pose to all-zero straight
        position_scale: unit scale for Manus positions (default meters)
    """
    config_path = get_default_config_path(robot_name, retargeting_type, hand_type)
    robot_dir = (
        Path(__file__).absolute().parent.parent.parent / "assets" / "robots" / "hands"
    )
    queue = multiprocessing.Queue(maxsize=10)
    producer_process = multiprocessing.Process(
        target=produce_frame, args=(queue, ros_topic, position_scale)
    )
    consumer_process = multiprocessing.Process(
        target=start_retargeting,
        args=(
            queue,
            str(robot_dir),
            str(config_path),
            publish_topic,
            publish_rate_hz,
            filter_type,
            filter_window,
            filter_sigma,
            ema_alpha,
            max_vel_rad,
            anti_flip_enable,
            anti_flip_min_rad,
            anti_flip_blend,
            anti_flip_strict,
            anti_flip_use_urdf_limits,
            anti_flip_lower_offset_rad,
            anti_flip_hard_limit,
            anti_flip_extra_joint4_fingers,
            anti_flip_fingers,
            anti_flip_seed_straight,
        ),
    )

    producer_process.start()
    logger.info("Started producer process")

    # 等待生产者初始化完成
    time.sleep(2)  # 给ROS2节点启动时间
    consumer_process.start()
    logger.info("Started consumer process")
    producer_process.join()
    consumer_process.join()
    time.sleep(5)

    print("done")

if __name__ == "__main__":
    tyro.cli(main)

"""
python manus_retargeting_ros2.py \
  --robot_name wuji \
  --retargeting_type dexpilot \
  --hand-type right \
  --ros-topic /manus_node_poses_0 \
  --publish-topic /hand_0/joint_commands \
  --publish-rate-hz 80 \
  --max-vel-rad 3.0 \
  --filter-type ema \
  --ema-alpha 0.35

"""
