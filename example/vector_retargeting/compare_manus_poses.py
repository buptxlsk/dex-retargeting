import time
from dataclasses import dataclass
from typing import Dict, Optional, List

import numpy as np
import tyro

import rclpy
from rclpy.node import Node
from manus_ros2_msgs.msg import ManusNodePoses


@dataclass
class Args:
    left_topic: str = "/manus_node_poses_left"
    right_topic: str = "/manus_node_poses_right"
    report_hz: float = 2.0
    output_path: Optional[str] = None


def _poses_to_array(msg: ManusNodePoses) -> Dict[int, np.ndarray]:
    pose_by_id = {}
    for node_id, pose in zip(msg.node_ids, msg.poses):
        pose_by_id[int(node_id)] = np.array(
            [pose.position.x, pose.position.y, pose.position.z], dtype=np.float32
        )
    return pose_by_id


class PoseComparer(Node):
    def __init__(self, args: Args):
        super().__init__("manus_pose_comparer")
        self.args = args
        self.left_latest: Optional[Dict[int, np.ndarray]] = None
        self.right_latest: Optional[Dict[int, np.ndarray]] = None
        self.samples_left: List[np.ndarray] = []
        self.samples_right: List[np.ndarray] = []
        self.timestamps: List[float] = []

        self.create_subscription(
            ManusNodePoses, args.left_topic, self._left_cb, 10
        )
        self.create_subscription(
            ManusNodePoses, args.right_topic, self._right_cb, 10
        )
        period = 1.0 / max(0.1, args.report_hz)
        self.create_timer(period, self._report)

        self.get_logger().info(
            f"Comparing {args.left_topic} vs {args.right_topic}"
        )

    def _left_cb(self, msg: ManusNodePoses) -> None:
        self.left_latest = _poses_to_array(msg)

    def _right_cb(self, msg: ManusNodePoses) -> None:
        self.right_latest = _poses_to_array(msg)

    def _report(self) -> None:
        if self.left_latest is None or self.right_latest is None:
            return

        common_ids = sorted(
            set(self.left_latest.keys()) & set(self.right_latest.keys())
        )
        if not common_ids:
            self.get_logger().warning("No common node_ids to compare.")
            return

        left_arr = np.stack([self.left_latest[i] for i in common_ids], axis=0)
        right_arr = np.stack([self.right_latest[i] for i in common_ids], axis=0)
        diff = left_arr - right_arr
        mean_abs = float(np.mean(np.abs(diff)))
        max_abs = float(np.max(np.abs(diff)))

        self.get_logger().info(
            f"Common nodes: {len(common_ids)} | mean_abs {mean_abs:.6f} | "
            f"max_abs {max_abs:.6f}"
        )

        if self.args.output_path:
            self.samples_left.append(left_arr)
            self.samples_right.append(right_arr)
            self.timestamps.append(time.time())

    def save(self) -> None:
        if not self.args.output_path or not self.timestamps:
            return
        np.savez(
            self.args.output_path,
            timestamps=np.array(self.timestamps, dtype=np.float64),
            left=np.stack(self.samples_left, axis=0),
            right=np.stack(self.samples_right, axis=0),
        )
        self.get_logger().info(f"Saved compare log to {self.args.output_path}")


def main() -> None:
    args = tyro.cli(Args)
    rclpy.init()
    node = PoseComparer(args)
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.save()
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
