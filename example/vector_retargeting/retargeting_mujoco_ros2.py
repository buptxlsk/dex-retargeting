import multiprocessing
import time
from pathlib import Path
from queue import Empty
from collections import deque
from typing import Optional, Dict

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


OPERATOR2MANO_RIGHT = np.array(
    [
        [0, 0, -1],
        [-1, 0, 0],
        [0, 1, 0],
    ]
)
OPERATOR2MANO_LEFT = np.array(
    [
        [0, 0, -1],
        [1, 0, 0],
        [0, -1, 0],
    ]
)

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


class ROS2LandmarkSubscriber(Node):
    """ROS2 节点，订阅手部关键点 PoseArray 消息，把 21x3 点塞进 queue。"""

    def __init__(
        self,
        queue: multiprocessing.Queue,
        topic_name: str,
        hand_type: HandType,
    ):
        super().__init__("hand_retargeting_node")
        self.queue = queue
        self.operator2mano = (
            OPERATOR2MANO_LEFT if hand_type == HandType.left else OPERATOR2MANO_RIGHT
        )
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

            # 5. 丢进队列（保留最新帧）
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
    """建立 MuJoCo 里 joint name -> qpos index 的映射（只管 hinge 关节）"""
    name_to_qpos = {}
    for j in range(model.njnt):
        jtype = model.jnt_type[j]
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
    filter_type: str,
    filter_window: int,
    filter_sigma: float,
    ema_alpha: float,
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

    # retargeting 侧 DOF 名称
    dof_names = list(retargeting.optimizer.robot.dof_joint_names)
    logger.info(f"Retargeting DOF joints: {dof_names}")
    target_joint_names = list(retargeting.optimizer.target_joint_names)
    idx_target_in_qpos = [dof_names.index(name) for name in target_joint_names]

    # --- init MuJoCo ---
    mj_model = mujoco.MjModel.from_xml_path(mjcf_path)
    mj_model.opt.gravity[:] = 0
    mj_data = mujoco.MjData(mj_model)
    name_to_qpos = build_mujoco_joint_mapping(mj_model)

    # 检查有没有找不到的关节
    missing = [n for n in dof_names if n not in name_to_qpos]
    if missing:
        logger.warning(
            f"The following retargeting DOF joints not found in MuJoCo model: {missing}"
        )

    mujoco.mj_forward(mj_model, mj_data)

    history = deque(maxlen=max(1, filter_window))
    gaussian_weights = None
    if filter_type == "gaussian":
        window_size = max(1, filter_window)
        if window_size % 2 == 0:
            window_size += 1
        history = deque(maxlen=window_size)
        gaussian_weights = _gaussian_weights(window_size, filter_sigma)

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
                for i, name in enumerate(target_joint_names)
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
                for i, name in enumerate(target_joint_names)
                if name == target_name
            ]
            if indices:
                finger_name_groups.setdefault(finger_id, []).extend(indices)

    joint_lower_bounds = {}
    if anti_flip_use_urdf_limits:
        for i, name in enumerate(target_joint_names):
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

    if anti_flip_seed_straight:
        last_good_qpos = np.zeros(len(target_joint_names), dtype=np.float32)
    else:
        last_good_qpos = None
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
                last = now

                # 拿 queue 里最新一帧关键点
                latest = None
                try:
                    while True:
                        latest = queue.get_nowait()
                except Empty:
                    pass

                joint_pos = latest if latest is not None else None

                if joint_pos is not None:
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
                        if retargeting_type == "DEXPILOT":
                            ref_value = _append_fingertip_direction(
                                ref_value, joint_pos, retargeting.optimizer
                            )

                    qpos = retargeting.retarget(ref_value).astype(np.float64)
                    if filter_type == "ema":
                        if len(history) == 0:
                            history.append(qpos)
                        else:
                            last = history[-1]
                            smoothed = ema_alpha * qpos + (1 - ema_alpha) * last
                            history.append(smoothed)
                        qpos = history[-1]
                    elif filter_type == "gaussian":
                        history.append(qpos)
                        if len(history) == history.maxlen:
                            stacked = np.stack(history, axis=0)
                            qpos = np.sum(
                                stacked * gaussian_weights[:, None], axis=0
                            )

                    if anti_flip_enable and finger_name_groups:
                        qpos_target = qpos[idx_target_in_qpos].copy()
                        if last_good_qpos is None:
                            last_good_qpos = qpos_target.copy()
                        abnormal_indices = []
                        for indices in finger_name_groups.values():
                            for idx in indices:
                                name = target_joint_names[idx]
                                lower = joint_lower_bounds.get(name, anti_flip_min_rad)
                                if qpos_target[idx] < lower:
                                    abnormal_indices.append(idx)
                        if abnormal_indices:
                            qpos_corrected = qpos_target.copy()
                            qpos_corrected[abnormal_indices] = (
                                (1.0 - anti_flip_blend)
                                * qpos_corrected[abnormal_indices]
                                + anti_flip_blend
                                * last_good_qpos[abnormal_indices]
                            )
                            if anti_flip_strict:
                                for idx in abnormal_indices:
                                    name = target_joint_names[idx]
                                    lower = joint_lower_bounds.get(
                                        name, anti_flip_min_rad
                                    )
                                    qpos_corrected[idx] = max(
                                        qpos_corrected[idx], lower
                                    )
                            qpos_target = qpos_corrected
                            qpos[idx_target_in_qpos] = qpos_target
                            retargeting.last_qpos = qpos_target.astype(np.float32)
                        else:
                            last_good_qpos = qpos_target.copy()

                    # 写回 MuJoCo 的 qpos（这里加拇指 offset）
                    for i, name in enumerate(dof_names):
                        if name not in name_to_qpos:
                            continue
                        adr = name_to_qpos[name]
                        val = qpos[i]
                        mj_data.qpos[adr] = val

                # 子步积分（有点物理感，虽然你重力关了）
                n_substeps = 5
                for _ in range(n_substeps):
                    mujoco.mj_step(mj_model, mj_data)

                v.sync()

    except KeyboardInterrupt:
        logger.info("Retargeting-MuJoCo loop interrupted by user.")
    finally:
        rclpy.shutdown()
        logger.info("ROS2 shutdown.")


def produce_frame(
    queue: multiprocessing.Queue,
    ros_topic: Optional[str] = None,
    hand_type: HandType = HandType.right,
):
    """ROS2 订阅进程：只负责往 queue 塞 21x3 关键点。"""
    if ros_topic is None:
        ros_topic = "/vrpn/hand_kp"

    rclpy.init()
    node = ROS2LandmarkSubscriber(queue, ros_topic, hand_type)
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
    filter_type: str = "none",
    filter_window: int = 5,
    filter_sigma: float = 1.0,
    ema_alpha: float = 0.2,
    anti_flip_enable: bool = True,
    anti_flip_min_rad: float = -0.2,
    anti_flip_blend: float = 0.7,
    anti_flip_strict: bool = True,
    anti_flip_use_urdf_limits: bool = True,
    anti_flip_lower_offset_rad: float = 0.3,
    anti_flip_hard_limit: bool = True,
    anti_flip_extra_joint4_fingers: str = "1,5",
    anti_flip_fingers: str = "1,2,3,4,5",
    anti_flip_seed_straight: bool = True,
):
    """
    retargeting_ros2 的 MuJoCo 版本：
    - 左进：ROS2 PoseArray (21x3 手部关键点)
    - 中间：DexPilot retargeting
    - 右出：MuJoCo qpos
    - filter_type: "none", "ema", or "gaussian"
    - anti_flip_*: anti-hyperextension correction (same as retargeting_ros2)
    """
    config_path = get_default_config_path(robot_name, retargeting_type, hand_type)
    robot_dir = (
        Path(__file__).absolute().parent.parent.parent / "assets" / "robots" / "hands"
    )

    queue = multiprocessing.Queue(maxsize=10)

    producer_process = multiprocessing.Process(
        target=produce_frame,
        args=(queue, ros_topic, hand_type),
    )
    consumer_process = multiprocessing.Process(
        target=start_retargeting_mujoco,
        args=(
            queue,
            str(robot_dir),
            str(config_path),
            mjcf_path,
            filter_type,
            filter_window,
            filter_sigma,
            ema_alpha,
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
    logger.info("Started ROS2 landmark producer process.")

    time.sleep(2)  # 给 ROS2 节点一点启动时间

    consumer_process.start()
    logger.info("Started retargeting + MuJoCo consumer process.")

    producer_process.join()
    consumer_process.join()

    logger.info("All processes finished.")


if __name__ == "__main__":
    tyro.cli(main)
