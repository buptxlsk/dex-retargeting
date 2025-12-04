import pandas as pd
import numpy as np
from pathlib import Path


FOLDER = "data_hotrack_test_2"
SEG_BEGIN = 100
SEG_END = 400


def load_landmarks(path):
    df = pd.read_csv(path)
    df["timestamp"] = df["stamp_sec"] + df["stamp_nanosec"] * 1e-9
    return df


def load_retarget(path):
    df = pd.read_csv(path)
    df["timestamp"] = df["time_sec"] + df["time_ns"] * 1e-9
    return df


def load_joint_commands(path):
    df = pd.read_csv(path)
    df_p = df[df["joint_name"] == "R_pinky_pip_joint"].copy()
    df_p["timestamp"] = df_p["stamp_sec"] + df_p["stamp_nanosec"] * 1e-9
    return df_p.sort_values("frame_id")


def compute_human_pinky_angle(df_lm_row):
    """Compute MCP–PIP–DIP bending angle"""
    p17 = df_lm_row[["p17_x", "p17_y", "p17_z"]].values.astype(float)
    p18 = df_lm_row[["p18_x", "p18_y", "p18_z"]].values.astype(float)
    p19 = df_lm_row[["p19_x", "p19_y", "p19_z"]].values.astype(float)

    v1 = p17 - p18
    v2 = p19 - p18

    if np.linalg.norm(v1) < 1e-8 or np.linalg.norm(v2) < 1e-8:
        return np.nan

    cos = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
    cos = np.clip(cos, -1.0, 1.0)
    return np.arccos(cos)


def main():
    folder = Path(FOLDER)

    lm = load_landmarks(folder/"hotrack_landmarks_log.csv")
    rt = load_retarget(folder/"retarget_internal_log.csv")
    jc = load_joint_commands(folder/"joint_commands_log.csv")

    # ---- 限定 frame_id 范围 ----
    rt_seg = rt[(rt["frame_id"] >= SEG_BEGIN) & (rt["frame_id"] <= SEG_END)]

    print(f"Segment frames found = {len(rt_seg)}")

    human_angles = []
    cmd_angles = []
    frame_ids = []

    for _, row in rt_seg.iterrows():
        frame_id = row["frame_id"]
        ts = row["timestamp"]

        # 找 hotrack 中最接近的帧
        idx = (lm["timestamp"] - ts).abs().idxmin()
        lm_row = lm.loc[idx]

        angle = compute_human_pinky_angle(lm_row)

        # 找 joint command 中同 frame 的指令（可能多行但 frame_id 排序好的）
        jc_rows = jc[jc["frame_id"] == frame_id]
        if len(jc_rows) > 0:
            cmd = float(jc_rows["position"].iloc[-1])
        else:
            cmd = np.nan

        human_angles.append(angle)
        cmd_angles.append(cmd)
        frame_ids.append(frame_id)

    human = np.array(human_angles)
    cmd = np.array(cmd_angles)

    print("\n=== HUMAN vs COMMAND (frame 100–400) ===")
    print("mean(human) =", np.nanmean(human))
    print("mean(cmd)   =", np.nanmean(cmd))
    print("mean(cmd - human) =", np.nanmean(cmd - human))
    print("corr =", np.corrcoef(human, cmd)[0, 1])

    # 检查异常：人很直但机器人弯
    abnormal = (human < 0.4) & (cmd > 1.0)
    idxs = np.where(abnormal)[0]
    print("\nABNORMAL FRAMES (human straight & cmd bent):")
    print("count =", len(idxs))
    print("frames =", [frame_ids[i] for i in idxs[:20]])


if __name__ == "__main__":
    main()
