#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import math
import rclpy
from rclpy.node import Node

from std_msgs.msg import String
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint

# 动作编码表
POSTURE_MAP = {"0": "walk", "1": "stand", "2": "sit", "3": "squat"}
GESTURE_MAP = {"a": "none", "b": "wave", "c": "reach"}

def decode_action(s: str):
    s = (s or "").strip().lower()
    if len(s) != 2:
        return None, None
    if s[0] not in POSTURE_MAP or s[1] not in GESTURE_MAP:
        return None, None
    return POSTURE_MAP[s[0]], GESTURE_MAP[s[1]]

class PlatformHeightController(Node):
    """
    控制平台升降，根据 /human_action 消息调整电机高度
    """

    def __init__(self):
        super().__init__("platform_height_controller_debug")

        # 发布 JointTrajectory
        self.traj_pub = self.create_publisher(
            JointTrajectory,
            "/gix_controller/joint_trajectory",
            10
        )

        # 订阅动作
        self.create_subscription(String, "/human_action", self.action_cb, 10)

        # 服务开关
        self.service = False

        # 高度(cm) -> 电机弧度(rad)
        self.RAD_40 = math.radians(0)
        self.RAD_60 = math.radians(180)
        self.RAD_90 = math.radians(360)

        self.move_sec = 2
        self.last_target_rad = None

        self.get_logger().info("平台升降控制节点启动 (DEBUG)")

        # 启动默认高度 40cm
        self.set_height_cm(40)

    def publish_rad(self, target_rad: float):
        traj = JointTrajectory()
        traj.joint_names = ["gix"]

        pt = JointTrajectoryPoint()
        pt.positions = [float(target_rad)]
        pt.time_from_start.sec = int(self.move_sec)
        pt.time_from_start.nanosec = 0

        traj.points = [pt]
        self.traj_pub.publish(traj)
        self.get_logger().info(f"[DEBUG] 已发布 JointTrajectory -> {target_rad:.3f} rad")

    def set_height_cm(self, cm: int):
        # cm -> rad
        if cm == 40:
            target = self.RAD_40
        elif cm == 60:
            target = self.RAD_60
        elif cm == 90:
            target = self.RAD_90
        else:
            self.get_logger().warn(f"[DEBUG] 不支持的高度：{cm}cm")
            return

        # 去重
        if self.last_target_rad is not None and abs(target - self.last_target_rad) < 1e-6:
            self.get_logger().info(f"[DEBUG] 目标与上次相同，跳过发布 {cm}cm")
            return

        self.last_target_rad = target
        self.publish_rad(target)
        self.get_logger().info(f"[DEBUG] 平台目标: {cm}cm -> {target:.3f} rad")

    def action_cb(self, msg: String):
        posture, gesture = decode_action(msg.data)
        self.get_logger().info(f"[DEBUG] 收到动作消息: {msg.data} -> posture={posture}, gesture={gesture}")

        if posture is None:
            self.get_logger().warn(f"[DEBUG] 动作格式错误: {msg.data}")
            return

        # wave 切换 service
        if gesture == "wave":
            self.service = not self.service
            self.get_logger().warn(f"[DEBUG] WAVE -> service={self.service}")
            if not self.service:
                self.get_logger().warn("[DEBUG] 结束服务 -> 平台40cm")
                self.set_height_cm(40)
                return
            else:
                self.get_logger().warn("[DEBUG] 开始服务 -> 根据姿态设置平台高度")

        if not self.service:
            self.set_height_cm(40)
            return

        # service=True 按姿态控制高度
        if posture in ("walk", "stand"):
            self.get_logger().warn("set to 90")
            self.set_height_cm(90)
        elif posture == "sit":
            self.get_logger().warn("set to 60")
            self.set_height_cm(60)
        elif posture == "squat":
            self.set_height_cm(40)
        else:
            self.set_height_cm(40)

def main():
    rclpy.init()
    node = PlatformHeightController()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == "__main__":
    main()