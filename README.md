# 简介
该模块使用dex-retargeting算法，能够根据PoseArray数据实现实时遥操omnihand

# 迁移过程
1. 在运行该项目之前，先将[原项目](https://github.com/dexsuite/dex-retargeting)中的\example跑通，包括使用POSITION以及VECTOR/DEXPILOT这两种方法

2. 解压omnihand_yuanshi，利用其得到需要的urdf文件并放入assets/robots/hands中

3. 注册新的机械手名称——omni
* 修改src/dex_retargeting/constants.py
* 根据规则由上面生成的urdf写出omni的yml文件，放入src/dex_retargeting/configs/teleop
* 再次运行示例中的命令，生成omni的仿真视频，注意每次注册新的机械手后都要重新pip install
```bash
cd ~/code/dex-retargeting
pip uninstall -y dex_retargeting
pip install -e .
```

# 环境
软件：environment.yml

硬件：
```text
+-----------------------------------------------------------------------------------------+
| NVIDIA-SMI 580.95.05              Driver Version: 580.95.05      CUDA Version: 13.0     |
+-----------------------------------------+------------------------+----------------------+
| GPU  Name                 Persistence-M | Bus-Id          Disp.A | Volatile Uncorr. ECC |
| Fan  Temp   Perf          Pwr:Usage/Cap |           Memory-Usage | GPU-Util  Compute M. |
|                                         |                        |               MIG M. |
|=========================================+========================+======================|
|   0  NVIDIA GeForce RTX 5060 Ti     Off |   00000000:01:00.0  On |                  N/A |
|  0%   42C    P5              8W /  180W |     933MiB /  16311MiB |      0%      Default |
|                                         |                        |                  N/A |
+-----------------------------------------+------------------------+----------------------+
```

# 直接运行

**1. 播放输入数据包**
```bash
  cd example/vector_retargeting/DATA_DEXRT/dexrt_vrpn_1
  ros2 bag play dexrt_vrpn_1_0.db3 -l 
```
**2. 启动节点输出**
```bash
  python retargeting_ros2.py \
  --robot_name omni \
  --retargeting_type dexpilot \
  --hand_type right \
  --ros_topic /hotrack/landmarks
```
此处ros_topic的名字取决于发布设备

**3. 新建终端检测输出**
```bash
  ros2 topic echo /joint_commands
```
该话题中msg格式如下
```text
std_msgs/Header header
string[] name
float64[] position
float64[] velocity
float64[] effort
----------------------
示例
header: 
    stamp: 
        sec: 1763004233 
        nanosec: 907039370 
    frame_id: omnihand_right_base
name: 
- R_thumb_roll_joint 
- R_index_abad_joint
...
position: 
- 0.4316634526107366
- -0.2102860283871998
...
velocity: [] 
effort: []
```

若正常显示，打通链路如下：
```mermaid
graph TD
    A[/vrpn/hand_kp：bag 播放出的 VRPN 手部 3D 关键点] --> B[ROS2LandmarkSubscriber：把 PoseArray → 21×3，归一化+旋转]
    B --> C[优化器（DexPilot/Vector）：吃 3D 向量 → 求解 omnihand 的关节角]
    C --> D[/joint_commands：按 SAPIEN 关节顺序发布关节角 JointState]
```

**4. 将输出转换为SAPIEN 3D 窗口中手部模型的运动**
```bash
  python retargeting_sapien_ros2.py \
  --robot_name omni \
  --retargeting_type dexpilot \
  --hand_type right \
  --ros_topic /vrpn/hand_kp
```
此处ros_topic的名字取决于发布设备

**5. 将输出转换为MUJOCO 3D 窗口中手部模型的运动**

由于智元官方没有给出omnihand的.xml文件，使用开源的方法[urdf2mjcf](https://github.com/kscalelabs/urdf2mjcf)和[教程](https://blog.csdn.net/weixin_44334573/article/details/146248011)将.urdf转换为.xml并放入运行目录下
```bash
python retargeting_mujoco_ros2.py \
    --robot_name omni \
    --retargeting_type dexpilot \
    --hand_type right \
    --mjcf_path ./omnihand_right.xml \
    --ros_topic /hotrack/landmarks
```
此处ros_topic的名字取决于发布设备