import pickle
from pathlib import Path

import cv2
import tqdm
import tyro

from dex_retargeting.constants import (
    RobotName,
    RetargetingType,
    HandType,
    get_default_config_path,
)
from dex_retargeting.retargeting_config import RetargetingConfig
from dex_retargeting.seq_retarget import SeqRetargeting
from single_hand_detector import SingleHandDetector


# -----------------------------
# 统一的 thumb offset（和 MuJoCo 那份一致）
# -----------------------------
THUMB_OFFSETS = {
    # "thumb_cmc_roll":   0.0,
    # "thumb_cmc_yaw":    0.0,
    # "thumb_cmc_pitch":  0.0,  
    "R_thumb_roll_joint": -0.191,
    "R_thumb_abad_joint": 0.0434,
    "R_thumb_mcp_joint": 0.912,
}


def apply_thumb_offset(qpos, dof_names):
    """给 retarget 出来的 qpos 加上 thumb offset."""
    qpos = qpos.copy()

    for jname, off in THUMB_OFFSETS.items():
        if jname in dof_names:
            idx = dof_names.index(jname)
            qpos[idx] += off
    return qpos


def retarget_video(
    retargeting: SeqRetargeting, video_path: str, output_path: str, config_path: str
):
    cap = cv2.VideoCapture(video_path)

    data = []

    if not cap.isOpened():
        print("Error: Could not open video file.")
        return

    detector = SingleHandDetector(hand_type="Right", selfie=False)
    length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    dof_names = retargeting.optimizer.robot.dof_joint_names

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

            # ----- Dexpilot retargeting logic -----
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

            qpos = retargeting.retarget(ref_value)

            # ----- Apply thumb offsets -----
            qpos = apply_thumb_offset(qpos, dof_names)

            data.append(qpos)
            pbar.update(1)

    # ===== 保存结果 =====
    meta_data = dict(
        config_path=config_path,
        dof=len(dof_names),
        joint_names=dof_names,
    )

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("wb") as f:
        pickle.dump(dict(data=data, meta_data=meta_data), f)
        print(f"joint_names: {meta_data['joint_names']}")

    cap.release()
    cv2.destroyAllWindows()

    retargeting.verbose()


def main(
    robot_name: RobotName,
    video_path: str,
    output_path: str,
    retargeting_type: RetargetingType,
    hand_type: HandType,
):
    config_path = get_default_config_path(robot_name, retargeting_type, hand_type)
    robot_dir = (
        Path(__file__).absolute().parent.parent.parent / "assets" / "robots" / "hands"
    )

    RetargetingConfig.set_default_urdf_dir(str(robot_dir))
    retargeting = RetargetingConfig.load_from_file(config_path).build()

    retarget_video(retargeting, video_path, output_path, str(config_path))


if __name__ == "__main__":
    tyro.cli(main)