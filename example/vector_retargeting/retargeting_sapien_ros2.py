import multiprocessing
import time
from pathlib import Path
from queue import Empty
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
from geometry_msgs.msg import PoseArray, Pose
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
    def __init__(self, robot_name):
        super().__init__('retargeting_joint_publisher')
        self.publisher_ = self.create_publisher(
            JointState, 
            '/joint_commands', 
            10
        )
        self.joint_names = []  # 将在后续填充
        self.robot_name = robot_name
        self.get_logger().info(f"JointState publisher created for {robot_name}")
        
    def publish_joints(self, positions):
        msg = JointState()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = self.robot_name + "_base"
        msg.name = self.joint_names
        msg.position = positions.tolist()  # 转换为Python list
        self.publisher_.publish(msg)

class ROS2LandmarkSubscriber(Node):
    """ROS2节点，订阅手部关键点PoseArray消息"""
    def __init__(self, queue: multiprocessing.Queue, topic_name: str):
        super().__init__('hand_retargeting_node')
        self.queue = queue
        self.operator2mano = (
            OPERATOR2MANO_RIGHT
        )
        self.subscription = self.create_subscription(
            PoseArray,
            topic_name,
            self.landmark_callback,
            10
        )
        logger.info(f"Subscribed to ROS2 topic: {topic_name}")

    def landmark_callback(self, msg: PoseArray):
        """处理PoseArray消息的回调函数"""
        try:
            # 解析21个关键点的位置
            joint_pos = np.zeros((21, 3))
            for i, pose in enumerate(msg.poses[:21]):  # 只取前21个关键点
                joint_pos[i] = [
                    pose.position.x / 1000,
                    pose.position.y / 1000,
                    pose.position.z / 1000
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
            logger.error(f"Error processing PoseArray: {str(e)}")

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

def start_retargeting(queue: multiprocessing.Queue, robot_dir: str, config_path: str):
    rclpy.init()
    RetargetingConfig.set_default_urdf_dir(str(robot_dir))
    logger.info(f"Start retargeting with config {config_path}")
    retargeting = RetargetingConfig.load_from_file(config_path).build()
    config = RetargetingConfig.load_from_file(config_path)

    # ---------- SAPIEN 场景 ----------
    sapien.render.set_viewer_shader_dir("default")
    sapien.render.set_camera_shader_dir("default")

    scene = sapien.Scene()

    # 地面 + 简单材质
    render_mat = sapien.render.RenderMaterial()
    render_mat.base_color = [0.06, 0.08, 0.12, 1]
    render_mat.metallic = 0.0
    render_mat.roughness = 0.9
    render_mat.specular = 0.8
    scene.add_ground(-0.2, render_material=render_mat, render_half_size=[1000, 1000])

    # 灯光（随便给两盏灯 + 环境光）
    scene.add_directional_light(np.array([1, 1, -1]), np.array([3, 3, 3]))
    scene.add_point_light(np.array([2, 2, 2]), np.array([2, 2, 2]), shadow=False)
    scene.set_environment_map(
        create_dome_envmap(sky_color=[0.2, 0.2, 0.2], ground_color=[0.2, 0.2, 0.2])
    )

    # 相机
    cam = scene.add_camera(
        name="Cheese!", width=600, height=600, fovy=1, near=0.1, far=10
    )
    cam.set_local_pose(sapien.Pose([0.5, 0, 0.0], [0, 0, 0, -1]))

    # ---------- 载入机器人 ----------
    loader = scene.create_urdf_loader()
    filepath = Path(config.urdf_path)
    robot_name = filepath.stem
    if "glb" not in robot_name:
        filepath = str(filepath).replace(".urdf", "_glb.urdf")
    else:
        filepath = str(filepath)
    robot = loader.load(filepath)
    joint_publisher_node = JointStatePublisher(robot_name)

    # 让手在相机前面稍微往下放一点
    robot.set_pose(sapien.Pose([0, 0, -0.05]))

    # ---------- Viewer ----------
    viewer = Viewer()
    viewer.set_scene(scene)
    viewer.control_window.show_origin_frame = False
    viewer.control_window.move_speed = 0.01
    viewer.control_window.toggle_camera_lines(False)
    viewer.set_camera_pose(cam.get_local_pose())

    # Different robot loader may have different orders for joints
    sapien_joint_names = [joint.get_name() for joint in robot.get_active_joints()]
    joint_publisher_node.joint_names = sapien_joint_names  # 设置发布者的关节名称
    retargeting_joint_names = retargeting.joint_names
    retargeting_to_sapien = np.array(
        [retargeting_joint_names.index(name) for name in sapien_joint_names]
    ).astype(int)
# -------------------------------------------------------------------
    while True:
        # 如果 viewer 关了，就退出循环
        if viewer.closed:
            break

        try:
            # 从队列获取关节数据
            joint_pos = queue.get(timeout=1.0)
            
            # 处理重定向
            retargeting_type = retargeting.optimizer.retargeting_type
            indices = retargeting.optimizer.target_link_human_indices
            
            if retargeting_type == "POSITION":
                ref_value = joint_pos[indices, :]
            else:
                origin_indices = indices[0, :]
                task_indices = indices[1, :]
                ref_value = joint_pos[task_indices, :] - joint_pos[origin_indices, :]

            qpos = retargeting.retarget(ref_value)
            mapped_qpos = qpos[retargeting_to_sapien]

            # ✅ 1) 让 SAPIEN 里的手动起来
            robot.set_qpos(mapped_qpos)
            scene.step()

            # ✅ 2) 发布关节状态给外部（MuJoCo / 真机）
            joint_publisher_node.publish_joints(mapped_qpos)
            
        except Empty:
            logger.warning("No hand data received in 1 second. Applying default pose.")

        # 渲染一帧 SAPIEN 画面
        viewer.render()

        # （可选）键盘 q 退出
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

        # 处理一下 ROS2 的事件
        rclpy.spin_once(joint_publisher_node, timeout_sec=0.001)

    # 清理
    joint_publisher_node.destroy_node()
    rclpy.shutdown()

def produce_frame(queue: multiprocessing.Queue, ros_topic: Optional[str] = None):
    if ros_topic is None:
        ros_topic = "/vrpn/hand_kp"  # 默认话题
    
    rclpy.init()
    pos_subscriber = ROS2LandmarkSubscriber(queue, ros_topic)
    
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
    ros_topic: str = "/vrpn/hand_kp" # 实际topic
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
    """
    config_path = get_default_config_path(robot_name, retargeting_type, hand_type)
    robot_dir = (
        Path(__file__).absolute().parent.parent.parent / "assets" / "robots" / "hands"
    )
    queue = multiprocessing.Queue(maxsize=10)
    producer_process = multiprocessing.Process(
        target=produce_frame, args=(queue, ros_topic)
    )
    consumer_process = multiprocessing.Process(
        target=start_retargeting, args=(queue, str(robot_dir), str(config_path))
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
