#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from pathlib import Path
import numpy as np
from sensor_msgs.msg import JointState
from manus_ros2_msgs.msg import ManusNodePoses
import torch

# 导入retarget相关模块
from dex_retargeting.constants import RobotName, HandType, RetargetingType, get_default_config_path
from dex_retargeting.retargeting_config import RetargetingConfig
from pd_realtime_0 import kabsch, apply_transform, DexHandFK

# 配准点和指尖点的节点ID定义
REGISTRATION_NODE_IDS = [0, 1, 6, 11, 16, 21]  # 手腕 + 5个指根
FINGERTIP_NODE_IDS = [0, 4, 9, 14, 19, 24]     # 手腕 + 5个指尖

# 机械手固定的配准点坐标
MANIP_REGISTRATION_POINTS = np.array([
    [0.0, 0.0, 0.0],           # 手腕
    [0.0327, 0.0239, 0.1071],  # 拇指根
    [0.0069, 0.0349, 0.1597],  # 食指根
    [0.007, 0.0106, 0.1687],   # 中指根
    [0.007, -0.0137, 0.1597],  # 无名指根
    [0.007, -0.038, 0.1457]    # 小指根
], dtype=np.float32)


class FingertipCalibrator:
    def __init__(self, device='cuda'):
        self.device = device
        self.with_scale = False
        self.allow_reflection = False
        self.manip_reg_points = MANIP_REGISTRATION_POINTS

    def calibrate(self, human_reg_points: np.ndarray, human_fingertips: np.ndarray) -> np.ndarray:
        R, t, s = kabsch(
            P=human_reg_points,
            Q=self.manip_reg_points,
            with_scale=self.with_scale,
            allow_reflection=self.allow_reflection
        )
        return apply_transform(human_fingertips, R, t, s)


class DirectRetargetInterface:
    """直接使用15对向量的Retarget接口"""
    def __init__(self, robot_name: RobotName, hand_type: HandType = HandType.right):
        self.robot_name = robot_name
        self.hand_type = hand_type
        self.retargeting_type = RetargetingType.dexpilot  # 使用DexPilot优化器
        
        # 初始化retargeting配置
        self._init_retargeting()
        self._init_joint_mapping()

    def _init_retargeting(self):
        """初始化重定向器"""
        RetargetingConfig.set_default_urdf_dir(
            str(Path("/home/cat/mk/retarget/neptune/dex_retargeting/assets/robots/hands"))
        )
        
        config_path = get_default_config_path(
            self.robot_name, 
            self.retargeting_type, 
            self.hand_type
        )
        
        print(f"Loading retargeting config from {config_path}")
        self.config = RetargetingConfig.load_from_file(config_path)
        self.retargeting = self.config.build()
        
        print(f"Retargeting optimizer type: {type(self.retargeting.optimizer).__name__}")

    def _init_joint_mapping(self):
        """初始化关节映射"""
        # 目标关节顺序
        self.target_joint_order = [
            'r_f_joint1_2', 'r_f_joint1_3', 'r_f_joint1_4',
            'r_f_joint2_2', 'r_f_joint2_3', 'r_f_joint2_4',
            'r_f_joint3_2', 'r_f_joint3_3', 'r_f_joint3_4',
            'r_f_joint4_2', 'r_f_joint4_3', 'r_f_joint4_4',
            'r_f_joint5_2', 'r_f_joint5_3', 'r_f_joint5_4',
            'r_f_joint1_1', 'r_f_joint2_1', 'r_f_joint3_1',
            'r_f_joint4_1', 'r_f_joint5_1'
        ]
        
        # retargeting关节名称
        self.retargeting_joint_names = self.retargeting.joint_names
        
        # 创建从retargeting到目标顺序的映射
        self.target_order_indices = [
            self.retargeting_joint_names.index(joint) 
            for joint in self.target_joint_order
        ]

    def retarget_from_vectors(self, vectors_15: np.ndarray) -> np.ndarray:
        """
        直接使用15对向量进行retarget
        :param vectors_15: 形状为(15, 3)的数组,包含15对向量
                          前10对: 指间向量
                          后5对: 手腕到指尖向量
        :return: 形状为(20,)的关节角度数组
        """
        if vectors_15.shape != (15, 3):
            raise ValueError(f"向量形状必须为(15,3),实际为{vectors_15.shape}")
        
        # vectors_10 = vectors_15[:10, :]
        # 直接调用retarget(输入为向量,不是位置坐标)
        qpos = self.retargeting.retarget(vectors_15)
        
        # 映射到目标关节顺序
        mapped_qpos = qpos[self.target_order_indices]
        
        return mapped_qpos


class JointStatePublisher(Node):
    def __init__(self, checkpoint_path):
        super().__init__('joint_state_publisher')

        # 1. 初始化直接retarget接口(使用15对向量)
        self.retarget_interface = DirectRetargetInterface(
            robot_name=RobotName.dexhand,
            hand_type=HandType.right
        )

        # 2. 初始化预测模型
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.predictor = self._load_model(checkpoint_path)

        # 3. ROS通信配置
        self.publisher = self.create_publisher(JointState, '/joint_commands', 10)
        self.subscription = self.create_subscription(
            ManusNodePoses,
            "/manus_node_poses_0",
            self.node_poses_callback,
            10
        )

        # 数据存储
        self.registration_points = {id: None for id in REGISTRATION_NODE_IDS}
        self.fingertip_coords = {id: None for id in FINGERTIP_NODE_IDS}
        self.latest_human_reg = None
        self.latest_human_ft = None

        # 发布配置
        self.publish_rate = 0.02  # 50Hz
        self.timer = self.create_timer(self.publish_rate, self.update_and_publish)
        
        # 关节顺序映射
        self.joint_order_map = {
            0:15, 1:0, 2:1, 3:2, 4:16, 5:3, 6:4, 7:5, 8:17, 9:6,
            10:7, 11:8, 12:18, 13:9, 14:10, 15:11, 16:19, 17:12, 18:13, 19:14
        }

        # 初始化配准器
        self.calibrator = FingertipCalibrator(device=self.device)
        
        # FK计算器(用于计算15对向量和验证指尖距离)
        self.fk_calculator = DexHandFK(
            xml_path='/home/cat/lc/RT/dexrobot_mujoco/dexrobot_mujoco/models/dexhand021_right.xml',
            device=self.device
        )
        
        self.print_counter = 0
        self.print_interval = 25

    def _load_model(self, checkpoint_path):
        """加载预测模型"""
        from train_pinch_dance_0 import JointLimitMLP
        checkpoint = torch.load(
            checkpoint_path, map_location=self.device, weights_only=True
        )
        model = JointLimitMLP(
            input_dim=45,
            output_dim=45,
            training=False
        ).to(self.device)
        model.load_state_dict(checkpoint["model_state_dict"])
        model.eval()
        self.get_logger().info(f"模型加载完成：{checkpoint_path}")
        return model

    def predict_vectors(self, input_vectors):
        """用模型预测优化后的15对向量"""
        input_data = input_vectors.flatten()
        input_tensor = torch.tensor(input_data, dtype=torch.float32).to(self.device).unsqueeze(0)
        
        with torch.no_grad():
            pred = self.predictor(input_tensor)
        
        return pred.squeeze().cpu().numpy().reshape(15, 3)

    def node_poses_callback(self, msg: ManusNodePoses):
        """处理配准点和指尖点"""
        if len(msg.node_ids) != len(msg.poses):
            self.get_logger().warn(f"node_ids与poses长度不匹配")
            return
        
        # 更新原始坐标
        for node_id, pose in zip(msg.node_ids, msg.poses):
            coord = np.array([pose.position.x, pose.position.y, pose.position.z], dtype=np.float32)
            
            if node_id in self.registration_points:
                self.registration_points[node_id] = coord
            
            if node_id in self.fingertip_coords:
                self.fingertip_coords[node_id] = coord
        
        # 检查数据完整性
        if not all(v is not None for v in self.registration_points.values()):
            return
        
        if not all(v is not None for v in self.fingertip_coords.values()):
            return
        
        # 整理原始坐标
        human_reg_points = np.stack([
            self.registration_points[id] for id in REGISTRATION_NODE_IDS
        ], axis=0)
        
        human_fingertips_raw = np.stack([
            self.fingertip_coords[id] for id in FINGERTIP_NODE_IDS
        ], axis=0)
        
        # 执行配准
        try:
            calibrated_coords = self.calibrator.calibrate(
                human_reg_points=human_reg_points,
                human_fingertips=human_fingertips_raw
            )
            
            self.latest_human_reg = human_reg_points
            self.latest_human_ft = calibrated_coords
            
        except Exception as e:
            self.get_logger().error(f"配准失败: {str(e)}")
            self.latest_human_ft = None

    def compute_fk_fingertip_distances(self, joint_angles_20):
        """
        通过FK计算优化后关节角度对应的指尖距离
        
        Args:
            joint_angles_20: (20,) 的关节角度数组
        
        Returns:
            distances_15: (15,) 的距离数组,包含10对指尖间距离 + 5对手腕到指尖距离
        """
        # 转换为torch张量并添加batch维度
        joint_tensor = torch.tensor(joint_angles_20, dtype=torch.float32, device=self.device).unsqueeze(0)  # (1, 20)
        
        # 计算指尖位置
        fingertip_positions = self.fk_calculator.compute_fingertip_positions(joint_tensor)  # (1, 5, 3)
        
        # 添加手腕坐标(原点)
        wrist_pos = torch.zeros(1, 1, 3, device=self.device)
        coords_with_wrist = torch.cat([wrist_pos, fingertip_positions], dim=1)  # (1, 6, 3)
        
        # 计算15对向量
        vectors_fk = self.fk_calculator.compute_15pair_vectors(coords_with_wrist)  # (1, 15, 3)
        
        # 计算向量的模(距离)
        distances = torch.norm(vectors_fk, dim=2).squeeze(0).cpu().numpy()  # (15,)
        
        return distances
    def _diagnose_dexpilot_projection(self, vectors_15):
        """诊断DexPilot投影机制"""
        # 计算拇指到其他四指的距离（前4对向量）
        thumb_distances = np.linalg.norm(vectors_15[:4], axis=1)
        
        project_dist = 0.03  # DexPilot默认投影阈值
        escape_dist = 0.05   # DexPilot默认退出阈值
        
        self.get_logger().info("=" * 80)
        self.get_logger().info("【DexPilot投影状态诊断】")
        self.get_logger().info(f"投影触发阈值: {project_dist*1000:.1f}mm, 退出阈值: {escape_dist*1000:.1f}mm")
        
        for i, name in enumerate(["拇指-食指", "拇指-中指", "拇指-无名指", "拇指-小指"]):
            dist_mm = thumb_distances[i] * 1000
            
            # 判断是否触发投影
            if dist_mm < project_dist * 1000:
                status = "🔴 已投影 (目标距离→0.1mm!)"
            elif dist_mm > escape_dist * 1000:
                status = "🟢 未投影"
            else:
                status = "🟡 滞后区"
            
            self.get_logger().info(f"{name}: {dist_mm:>6.2f}mm  {status}")
        
        self.get_logger().info("=" * 80)


    def update_and_publish(self):
        """执行流程: 15对向量计算→模型预测→直接retarget→FK验证→发布"""
        
        if self.latest_human_ft is None:
            self.get_logger().debug("等待校准后的坐标数据...")
            return
        
        if self.latest_human_ft.shape != (6, 3):
            self.get_logger().error(f"校准后坐标形状错误")
            return

        try:
            # 步骤1: 计算15对向量(从校准后的坐标)
            calibrated_vectors = self.fk_calculator.compute_15pair_vectors(
                torch.tensor(self.latest_human_ft).unsqueeze(0)
            ).squeeze(0).numpy()
            

            
            # 步骤2: 模型预测优化后的向量
            distances_calibrated = np.linalg.norm(calibrated_vectors, axis=1)
            pred_vectors = self.predict_vectors(calibrated_vectors)
            distances_predicted = np.linalg.norm(pred_vectors, axis=1)

            # self._diagnose_dexpilot_projection(pred_vectors)

            calib_dirs = calibrated_vectors / (np.linalg.norm(calibrated_vectors, axis=1, keepdims=True) + 1e-8)
            pred_dirs = pred_vectors / (np.linalg.norm(pred_vectors, axis=1, keepdims=True) + 1e-8)
            cosine_sim = np.sum(calib_dirs * pred_dirs, axis=1)  # 余弦相似度

            # self.get_logger().info("=== 向量方向诊断 ===")
            # for i in range(4):  # 只看前4对拇指向量
            #     self.get_logger().info(
            #         f"拇指向量{i}: 余弦相似度={cosine_sim[i]:.4f}, "
            #         f"角度偏差={np.degrees(np.arccos(np.clip(cosine_sim[i], -1, 1))):.2f}°"
            #     )

            # 步骤3: 使用预测向量进行retarget
            robot_joints = self.retarget_interface.retarget_from_vectors(calibrated_vectors)
            
            # ✅ 步骤4: 通过FK计算retarget后的指尖距离
            distances_fk = self.compute_fk_fingertip_distances(robot_joints)
            
            # 步骤5: 打印对比(每隔一段时间)
            self.print_counter += 1
            if self.print_counter >= self.print_interval:
                self.print_counter = 0
                self._log_comprehensive_comparison(
                    distances_calibrated, 
                    distances_predicted, 
                    distances_fk
                )

            # 步骤6: 发布关节角度
            self.publish_joint_angles(robot_joints)

        except Exception as e:
            self.get_logger().error(f"处理失败：{str(e)}")
            import traceback
            self.get_logger().error(traceback.format_exc())

    def _log_comprehensive_comparison(self, dist_calibrated, dist_predicted, dist_fk):
        """
        打印完整对比:校准距离 vs 预测距离 vs FK计算距离
        
        Args:
            dist_calibrated: 校准后的向量距离 (15,)
            dist_predicted: 神经网络预测后的向量距离 (15,)
            dist_fk: retarget优化后通过FK计算的指尖距离 (15,)
        """
        finger_pairs = [
            "拇指-食指", "拇指-中指", "拇指-无名指", "拇指-小指",
            "食指-中指", "食指-无名指", "食指-小指",
            "中指-无名指", "中指-小指", "无名指-小指",
            "手腕→拇指", "手腕→食指", "手腕→中指", "手腕→无名指", "手腕→小指"
        ]
        
        self.get_logger().info("=" * 100)
        self.get_logger().info("【完整距离对比】校准后 → 预测后 → FK验证(单位:米)")
        self.get_logger().info("-" * 100)
        self.get_logger().info(
            f"{'向量类型':<6} {'配对名称':<6} "
            f"{'校准距离':>6} {'预测距离':>6} {'FK距离':>10} "
            f"{'预测变化':>12} {'FK变化':>12}"
        )
        self.get_logger().info("-" * 100)
        
        for i, pair_name in enumerate(finger_pairs):
            vector_type = "指间" if i < 10 else "腕尖"
            
            # 计算变化量和百分比
            pred_diff = dist_predicted[i] - dist_calibrated[i]
            pred_percent = (pred_diff / dist_calibrated[i]) * 100 if dist_calibrated[i] != 0 else 0
            
            fk_diff = dist_fk[i] - dist_calibrated[i]
            fk_percent = (fk_diff / dist_calibrated[i]) * 100 if dist_calibrated[i] != 0 else 0
            
            # 添加颜色标记(如果变化过大)
            pred_flag = "⚠️" if abs(pred_percent) > 10 else "  "
            fk_flag = "⚠️" if abs(fk_percent) > 10 else "  "
            
            self.get_logger().info(
                f"{vector_type:<6} {pair_name:<10} "
                f"{dist_calibrated[i]:>10.4f} {dist_predicted[i]:>10.4f} {dist_fk[i]:>10.4f} "
                f"{pred_diff:>+6.4f}({pred_percent:>+5.1f}%){pred_flag} "
                f"{fk_diff:>+6.4f}({fk_percent:>+5.1f}%){fk_flag}"
            )
        
        # 统计信息
        self.get_logger().info("-" * 100)
        
        # 预测误差统计
        pred_errors = np.abs(dist_predicted - dist_calibrated)
        self.get_logger().info(
            f"【预测误差】平均={np.mean(pred_errors):.4f}m, "
            f"最大={np.max(pred_errors):.4f}m, "
            f"标准差={np.std(pred_errors):.4f}m"
        )
        
        # FK误差统计
        fk_errors = np.abs(dist_fk - dist_calibrated)
        self.get_logger().info(
            f"【FK误差】  平均={np.mean(fk_errors):.4f}m, "
            f"最大={np.max(fk_errors):.4f}m, "
            f"标准差={np.std(fk_errors):.4f}m"
        )
        
        # 预测与FK的一致性
        consistency_errors = np.abs(dist_fk - dist_predicted)
        self.get_logger().info(
            f"【预测-FK一致性】平均差异={np.mean(consistency_errors):.4f}m, "
            f"最大差异={np.max(consistency_errors):.4f}m"
        )
        
        self.get_logger().info("=" * 100)

    def publish_joint_angles(self, robot_joints):
        """发布关节角度"""
        msg = JointState()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = ''
        msg.name = [
            "r_f_joint1_1","r_f_joint1_2","r_f_joint1_3","r_f_joint1_4",
            "r_f_joint2_1","r_f_joint2_2","r_f_joint2_3","r_f_joint2_4",
            "r_f_joint3_1","r_f_joint3_2","r_f_joint3_3","r_f_joint3_4",
            "r_f_joint4_1","r_f_joint4_2","r_f_joint4_3","r_f_joint4_4",
            "r_f_joint5_1","r_f_joint5_2","r_f_joint5_3","r_f_joint5_4"
        ]
        msg.position = [robot_joints[self.joint_order_map[i]] for i in range(20)]
        self.publisher.publish(msg)


def main(args=None):
    rclpy.init(args=args)
    checkpoint_path = "checkpoints_5_fingers/20251203_153816_vector_best/best_model.pth"
    publisher = JointStatePublisher(checkpoint_path)
    
    try:
        rclpy.spin(publisher)
    except KeyboardInterrupt:
        pass
    finally:
        publisher.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()

# 向量类型   配对名称     校准距离   预测距离       FK距离         预测变化         FK变化
# 指间     拇指-食指          0.1123     0.1849     0.0831 +0.0726(+64.6%)⚠️ -0.0292(-26.0%)⚠️
# 指间     拇指-中指          0.1460     0.1610     0.0938 +0.0150(+10.3%)⚠️ -0.0521(-35.7%)⚠️
# 指间     拇指-无名指         0.1380     0.2163     0.1402 +0.0783(+56.7%)⚠️ +0.0021( +1.5%)  
# 指间     拇指-小指          0.1418     0.1891     0.1503 +0.0473(+33.3%)⚠️ +0.0085( +6.0%)  
# 指间     食指-中指          0.0423     0.0493     0.0486 +0.0070(+16.5%)⚠️ +0.0063(+14.8%)⚠️
# 指间     食指-无名指         0.0535     0.0993     0.1176 +0.0458(+85.7%)⚠️ +0.0641(+119.9%)⚠️
# 指间     食指-小指          0.0741     0.1190     0.1465 +0.0450(+60.7%)⚠️ +0.0725(+97.9%)⚠️
# 指间     中指-无名指         0.0358     0.0904     0.0822 +0.0546(+152.6%)⚠️ +0.0464(+129.6%)⚠️
# 指间     中指-小指          0.0517     0.0798     0.0995 +0.0281(+54.3%)⚠️ +0.0477(+92.3%)⚠️
# 指间     无名指-小指         0.0301     0.0731     0.0757 +0.0430(+142.6%)⚠️ +0.0455(+151.2%)⚠️
# 腕尖     手腕→拇指          0.1276     0.1924     0.1685 +0.0649(+50.9%)⚠️ +0.0410(+32.1%)⚠️
# 腕尖     手腕→食指          0.1830     0.2139     0.2491 +0.0309(+16.9%)⚠️ +0.0661(+36.1%)⚠️
# 腕尖     手腕→中指          0.1925     0.2289     0.2484 +0.0364(+18.9%)⚠️ +0.0560(+29.1%)⚠️
# 腕尖     手腕→无名指         0.1878     0.2721     0.2767 +0.0842(+44.8%)⚠️ +0.0889(+47.3%)⚠️
# 腕尖     手腕→小指          0.1702     0.2421     0.2528 +0.0719(+42.2%)⚠️ +0.0826(+48.5%)⚠️

