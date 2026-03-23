#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import math
import rclpy
from rclpy.node import Node

from std_msgs.msg import String
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint

# -----------------------------
# Action Encoding Table
# -----------------------------
POSTURE_MAP = {"0": "walk", "1": "stand", "2": "sit", "3": "squat"}
GESTURE_MAP = {"a": "none", "b": "wave", "c": "reach"}

def decode_action(s: str):
    s = (s or "").strip().lower()
    if len(s) != 2:
        return None, None
    if s[0] not in POSTURE_MAP or s[1] not in GESTURE_MAP:
        return None, None
    return POSTURE_MAP[s[0]], GESTURE_MAP[s[1]]


# -----------------------------
# ROS2 Node
# -----------------------------
class PlatformHeightController(Node):

    def __init__(self):
        super().__init__("platform_height_controller")

        # Control Publisher
        self.traj_pub = self.create_publisher(
            JointTrajectory,
            "/gix_controller/joint_trajectory",
            10
        )

        # Action Subscription
        self.create_subscription(String, "/human_action", self.action_cb, 10)

        # -----------------------------
        # State Control
        # -----------------------------
        self.started = False   # Activation flag (only allows single wave trigger)
        self.current_state = "SIT"  # Default initial state

        # -----------------------------
        # Height Mapping
        # -----------------------------
        self.RAD_40 = math.radians(0)
        self.RAD_90 = math.radians(360)

        self.move_sec = 2
        self.last_target_rad = None

        self.get_logger().info("Platform Control Node Started (Single Trigger Version)")

        # Initial height: 40cm
        self.set_height_cm(40)

    # -----------------------------
    # Send Trajectory
    # -----------------------------
    def publish_rad(self, target_rad: float):
        traj = JointTrajectory()
        traj.joint_names = ["gix"]

        pt = JointTrajectoryPoint()
        pt.positions = [float(target_rad)]
        pt.time_from_start.sec = int(self.move_sec)

        traj.points = [pt]
        self.traj_pub.publish(traj)

        self.get_logger().info(f"[MOVE] -> {target_rad:.3f} rad")

    # -----------------------------
    # Set Height
    # -----------------------------
    def set_height_cm(self, cm: int):

        if cm == 40:
            target = self.RAD_40
        elif cm == 90:
            target = self.RAD_90
        else:
            self.get_logger().warn(f"Unsupported height: {cm}")
            return

        # Deduplication (avoid redundant commands)
        if self.last_target_rad is not None and abs(target - self.last_target_rad) < 1e-6:
            return

        self.last_target_rad = target
        self.publish_rad(target)

        self.get_logger().warn(f"[HEIGHT] {cm}cm")

    # -----------------------------
    # Action Callback
    # -----------------------------
    def action_cb(self, msg: String):
        posture, gesture = decode_action(msg.data)

        self.get_logger().info(f"[ACTION] {msg.data} -> {posture}, {gesture}")

        if posture is None:
            return

        # =============================
        # 1️Only allow activation via the first wave
        # =============================
        if not self.started:
            if gesture == "wave":
                self.started = True
                self.get_logger().warn("System Activated (Triggered once)")
                self.set_height_cm(40)  # Initial sit height
            return

        # =============================
        # 2️Logic after activation
        # =============================
        # Ignore all subsequent waves
        # Only handle sit / stand / walk

        if posture == "sit":
            if self.current_state != "SIT":
                self.current_state = "SIT"
                self.set_height_cm(40)

        elif posture == "stand":
            if self.current_state != "STAND":
                self.current_state = "STAND"
                self.set_height_cm(90)
        
        elif posture == "walk":
            if self.current_state != "WALK":
                self.current_state = "WALK"
                self.set_height_cm(90)
        else:
            self.get_logger().info("[IGNORE] Non sit/stand/walk posture, no action taken")


# -----------------------------
# Main
# -----------------------------
def main():
    rclpy.init()
    node = PlatformHeightController()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()