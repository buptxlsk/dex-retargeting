#!/usr/bin/env python3
import argparse
import pickle
import numpy as np


def load_pkl(path):
    print(f"\n=== Loading {path} ===")
    with open(path, "rb") as f:
        data = pickle.load(f)
    print("Top-level keys:", list(data.keys()))
    return data


def ensure_array(x, name):
    x = np.asarray(x)
    if x.ndim == 2:
        # [T, 3] → 当成单指
        x = x[:, None, :]
    if x.ndim != 3 or x.shape[-1] != 3:
        raise ValueError(f"{name} expected shape [T, F, 3], got {x.shape}")
    return x


def compute_stats(human_vec, robot_vec):
    """
    human_vec, robot_vec: [T, F, 3]
    返回:
      per_finger: dict(finger_idx -> dict(mean, std, max))
      overall: dict(mean, std, max)
    """
    # 对齐长度（理论上两边 T 一样，这里防御性写法）
    T = min(human_vec.shape[0], robot_vec.shape[0])
    human_vec = human_vec[:T]
    robot_vec = robot_vec[:T]

    diff = robot_vec - human_vec       # [T, F, 3]
    err = np.linalg.norm(diff, axis=-1)  # [T, F]

    per_finger = {}
    F = err.shape[1]
    for fi in range(F):
        e = err[:, fi]
        per_finger[fi] = {
            "mean": float(e.mean()),
            "std": float(e.std()),
            "max": float(e.max()),
        }

    overall = {
        "mean": float(err.mean()),
        "std": float(err.std()),
        "max": float(err.max()),
    }
    return per_finger, overall


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--unified", type=str, required=True,
                        help="pkl from scaling=1 run, e.g. data/test_nor.pkl")
    parser.add_argument("--per_finger", type=str, required=True,
                        help="pkl from per-finger scaling run, e.g. data/test_per.pkl")
    parser.add_argument("--finger-names", type=str, nargs="*", default=None,
                        help="Optional: names for fingers in order, e.g. thumb index middle ring pinky")
    parser.add_argument("--human-key", type=str, default="human_vec",
                        help="Top-level key for human vectors")
    parser.add_argument("--robot-key", type=str, default="robot_vec",
                        help="Top-level key for robot vectors")
    args = parser.parse_args()

    # 1. 载入两个 pkl
    data_uni = load_pkl(args.unified)
    data_per = load_pkl(args.per_finger)

    human_key = args.human_key
    robot_key = args.robot_key

    # 2. 检查 key 是否存在
    for tag, d in [("unified", data_uni), ("per_finger", data_per)]:
        if human_key not in d or robot_key not in d:
            raise KeyError(
                f"[{tag}] cannot find keys '{human_key}'/'{robot_key}'. "
                f"Available keys: {list(d.keys())}"
            )

    human_uni = ensure_array(data_uni[human_key], f"{human_key} (unified)")
    robot_uni = ensure_array(data_uni[robot_key], f"{robot_key} (unified)")
    human_per = ensure_array(data_per[human_key], f"{human_key} (per_finger)")
    robot_per = ensure_array(data_per[robot_key], f"{robot_key} (per_finger)")

    # 3. 计算误差
    per_uni, overall_uni = compute_stats(human_uni, robot_uni)
    per_per, overall_per = compute_stats(human_per, robot_per)

    F = human_uni.shape[1]
    finger_names = args.finger_names
    if finger_names is None or len(finger_names) != F:
        finger_names = [f"finger_{i}" for i in range(F)]

    print("\n================ OVERALL ERROR =================")
    print(f"Unified scaling : mean={overall_uni['mean']:.6f}, std={overall_uni['std']:.6f}, max={overall_uni['max']:.6f}")
    print(f"Per-finger sc.  : mean={overall_per['mean']:.6f}, std={overall_per['std']:.6f}, max={overall_per['max']:.6f}")
    ratio = overall_per["mean"] / overall_uni["mean"] if overall_uni["mean"] > 0 else np.nan
    print(f"Per / Unified mean ratio = {ratio:.3f}  (<1 表示 per-finger 更好)")

    print("\n================ PER-FINGER ERROR =================")
    header = f"{'finger':>10} | {'uni_mean':>10} | {'per_mean':>10} | {'ratio':>8} | {'uni_max':>10} | {'per_max':>10}"
    print(header)
    print("-" * len(header))
    for fi in range(F):
        name = finger_names[fi]
        eu = per_uni[fi]["mean"]
        ep = per_per[fi]["mean"]
        ru = per_uni[fi]["max"]
        rp = per_per[fi]["max"]
        r  = ep / eu if eu > 0 else np.nan
        print(f"{name:>10} | {eu:10.6f} | {ep:10.6f} | {r:8.3f} | {ru:10.6f} | {rp:10.6f}")

    print("\nDone ✅")


if __name__ == "__main__":
    main()
