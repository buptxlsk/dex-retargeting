#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState
import csv
from pathlib import Path


class JointCommandCSV(Node):
    def __init__(self, save_path="joint_commands.csv"):
        super().__init__("joint_command_csv")

        self.sub = self.create_subscription(
            JointState, "/joint_commands", self.callback, 10
        )

        self.save_path = Path(save_path)
        self.csv_file = open(self.save_path, "w", newline="")
        self.writer = None

        self.get_logger().info(f"Recording /joint_commands → {self.save_path}")

    def callback(self, msg: JointState):
        stamp = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9

        if self.writer is None:
            header = ["stamp"] + list(msg.name)
            self.writer = csv.writer(self.csv_file)
            self.writer.writerow(header)

        row = [stamp] + list(msg.position)
        self.writer.writerow(row)

    def destroy_node(self):
        super().destroy_node()
        print(f"\nSaved CSV to {self.save_path}")
        self.csv_file.close()


def main():
    rclpy.init()
    node = JointCommandCSV()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()

