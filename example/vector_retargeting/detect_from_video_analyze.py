import pickle
from pathlib import Path

import cv2
import tqdm
import tyro
import numpy as np  # 用于处理向量

from dex_retargeting.constants import (
    RobotName,
    RetargetingType,
    HandType,
    get_default_config_path,
)
from dex_retargeting.retargeting_config import RetargetingConfig
from dex_retargeting.seq_retarget import SeqRetargeting
from single_hand_detector import SingleHandDetector


def retarget_video(
    retargeting: SeqRetargeting, video_path: str, output_path: str, config_path: str
):
    cap = cv2.VideoCapture(video_path)

    qpos_list = []
    human_vec_list = []   # 分析用 human task-space 向量
    robot_vec_list = []   # 分析用 robot task-space 向量

    if not cap.isOpened():
        print("Error: Could not open video file.")
        return

    # 读一遍 config，用来拿 DexPilot 的 wrist_link_name / finger_tip_link_names
    cfg = RetargetingConfig.load_from_file(config_path)

    detector = SingleHandDetector(hand_type="Right", selfie=False)
    length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    optimizer = retargeting.optimizer
    retargeting_type = optimizer.retargeting_type  # "VECTOR" / "POSITION" / "DEXPILOT" 等字符串

    # config 里给的 human 索引（对 VECTOR / DEXPILOT 都是 2×N 的 origin/task 索引）
    indices = getattr(optimizer, "target_link_human_indices", None)
    robot = optimizer.robot

    has_vector_struct = (
        hasattr(optimizer, "computed_link_indices")
        and hasattr(optimizer, "origin_link_indices")
        and hasattr(optimizer, "task_link_indices")
    )

    with tqdm.tqdm(total=length) as pbar:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            rgb = frame[..., ::-1]
            num_box, joint_pos, keypoint_2d, mediapipe_wrist_rot = detector.detect(rgb)
            if num_box == 0:
                pbar.update(1)
                continue

            # joint_pos: [21, 3]

            # =========================
            # 1) 构造喂给 retarget() 的 ref_value（严格按方法原始定义）
            # =========================
            if retargeting_type == "POSITION":
                if indices is None:
                    raise ValueError("POSITION retargeting requires target_link_human_indices.")
                ref_value = joint_pos[indices, :]
            else:
                # VECTOR / DEXPILOT / 其他 vector-based 方法：origin→task
                if indices is None:
                    raise ValueError(f"{retargeting_type} retargeting requires target_link_human_indices.")
                origin_indices = indices[0, :]
                task_indices = indices[1, :]
                ref_value = joint_pos[task_indices, :] - joint_pos[origin_indices, :]  # [N_vec, 3]

            # =========================
            # 2) 调用 retarget 得到 qpos
            # =========================
            qpos = retargeting.retarget(ref_value)

            # =========================
            # 3) 计算“分析用”的 human_vec / robot_vec
            #    （不要求和优化内部完全一样，只要定义自洽，用来做误差对比）
            # =========================
            human_vec = None
            robot_vec = None

            # --- human_vec 定义 ---
            if retargeting_type == "VECTOR":
                # 对 vector 方法：直接用 ref_value 作为分析向量（origin→task）
                human_vec = ref_value.copy()
            elif retargeting_type == "DEXPILOT":
                # 对 DexPilot：分析用 wrist→五指尖
                # MediaPipe: wrist=0, thumb_tip=4, index_tip=8, middle_tip=12, ring_tip=16, pinky_tip=20
                wrist = joint_pos[0:1, :]                     # [1,3]
                tips = joint_pos[[4, 8, 12, 16, 20], :]       # [5,3]
                human_vec = tips - wrist                      # [5,3]
            else:
                human_vec = None

            # --- robot_vec 定义 ---
            if retargeting_type == "VECTOR" and has_vector_struct:
                # 和 VectorOptimizer objective 里的 FK 一致
                robot.compute_forward_kinematics(qpos)

                link_poses = [
                    robot.get_link_pose(index)
                    for index in optimizer.computed_link_indices
                ]
                body_pos = np.array([pose[:3, 3] for pose in link_poses])  # [L,3]

                origin_link_indices = optimizer.origin_link_indices.cpu().numpy()
                task_link_indices = optimizer.task_link_indices.cpu().numpy()

                origin_pos = body_pos[origin_link_indices, :]  # [F,3]
                task_pos = body_pos[task_link_indices, :]      # [F,3]

                robot_vec = task_pos - origin_pos              # [F,3]

            elif retargeting_type == "DEXPILOT":
                # DexPilot：用 config 里的 wrist_link_name + finger_tip_link_names 做 wrist→tips
                robot.compute_forward_kinematics(qpos)

                wrist_link_name = cfg.wrist_link_name
                finger_tip_link_names = cfg.finger_tip_link_names  # list[str]，顺序应为 thumb→pinky

                if wrist_link_name is None or finger_tip_link_names is None:
                    raise ValueError(
                        "DexPilot config must define wrist_link_name and finger_tip_link_names "
                        "to compute analysis robot_vec."
                    )

                wrist_index = robot.get_link_index(wrist_link_name)
                tip_indices = [robot.get_link_index(name) for name in finger_tip_link_names]

                wrist_pose = robot.get_link_pose(wrist_index)
                wrist_pos = wrist_pose[:3, 3]  # [3]

                tip_poses = [robot.get_link_pose(i) for i in tip_indices]
                tip_pos = np.array([p[:3, 3] for p in tip_poses])  # [5,3]

                robot_vec = tip_pos - wrist_pos[None, :]  # [5,3]

            # =========================
            # 4) 存这一帧
            # =========================
            qpos_list.append(qpos)
            human_vec_list.append(human_vec)
            robot_vec_list.append(robot_vec)

            pbar.update(1)

    # =========================
    # 5) 打包保存 pkl
    # =========================
    meta_data = dict(
        config_path=config_path,
        dof=len(retargeting.optimizer.robot.dof_joint_names),
        joint_names=retargeting.optimizer.robot.dof_joint_names,
        retargeting_type=retargeting_type,
    )

    qpos_array = np.asarray(qpos_list)

    if all(v is not None for v in human_vec_list):
        human_vec_array = np.stack(human_vec_list, axis=0)  # [T, F, 3]
    else:
        human_vec_array = None

    if all(v is not None for v in robot_vec_list):
        robot_vec_array = np.stack(robot_vec_list, axis=0)  # [T, F, 3]
    else:
        robot_vec_array = None

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    out_dict = dict(
        data=qpos_array,
        meta_data=meta_data,
    )
    if human_vec_array is not None:
        out_dict["human_vec"] = human_vec_array
    if robot_vec_array is not None:
        out_dict["robot_vec"] = robot_vec_array

    with output_path.open("wb") as f:
        pickle.dump(out_dict, f)

    retargeting.verbose()
    cap.release()
    cv2.destroyAllWindows()


def main(
    robot_name: RobotName,
    video_path: str,
    output_path: str,
    retargeting_type: RetargetingType,
    hand_type: HandType,
):
    """
    Detects the human hand pose from a video and translates the human pose trajectory into a robot pose trajectory.

    Args:
        robot_name: The identifier for the robot. This should match one of the default supported robots.
        video_path: The file path for the input video in .mp4 format.
        output_path: The file path for the output data in .pickle format.
        retargeting_type: The type of retargeting, each type corresponds to a different retargeting algorithm.
        hand_type: Specifies which hand is being tracked, either left or right.
            Please note that retargeting is specific to the same type of hand: a left robot hand can only be retargeted
            to another left robot hand, and the same applies for the right hand.
    """

    config_path = get_default_config_path(robot_name, retargeting_type, hand_type)
    robot_dir = (
        Path(__file__).absolute().parent.parent.parent / "assets" / "robots" / "hands"
    )
    RetargetingConfig.set_default_urdf_dir(str(robot_dir))
    retargeting = RetargetingConfig.load_from_file(config_path).build()
    retarget_video(retargeting, video_path, output_path, str(config_path))


if __name__ == "__main__":
    tyro.cli(main)
