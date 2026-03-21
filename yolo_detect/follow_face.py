#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Version 1 – Always face human (target_drift=0)
Full debug logging edition
"""

import rclpy
from rclpy.node import Node
import numpy as np
import cv2
from sensor_msgs.msg import CompressedImage
from geometry_msgs.msg import Twist
from std_msgs.msg import String
from sensor_msgs.msg import LaserScan
from ultralytics import YOLO

POSTURE_MAP = {"0": "walk", "1": "stand", "2": "sit", "3": "squat"}
GESTURE_MAP = {"a": "none", "b": "wave", "c": "reach"}


def decode_action(s: str):
    s = (s or "").strip().lower()
    if len(s) != 2:
        return None, None
    if s[0] not in POSTURE_MAP or s[1] not in GESTURE_MAP:
        return None, None
    return POSTURE_MAP[s[0]], GESTURE_MAP[s[1]]


class FollowerFSMV1(Node):

    def __init__(self):
        super().__init__("follower_fsm_v1_face_only")

        # ---------------- Parameters ----------------
        self.debug = True
        self.W = 640
        self.H = 480

        self.deadband = 40
        self.k_ang = 0.002

        self.bw_30 = 320.0
        self.bw_60 = 220.0
        self.bw_stop = 360.0

        # ---------------- FSM ----------------
        self.state = "IDLE"
        self.service = False

        self.posture = None
        self.gesture = None
        self.prev_posture = None

        self.prev_area = None
        self.scan = None
        self.lost_count = 0
        self.lost_threshold = 5   # 连续5帧没人才切换
        
        self.create_subscription(
            LaserScan,
            "/scan",
            self.scan_callback,
            10
        )
        # ---------------- ROS ----------------
        self.create_subscription(
            CompressedImage, "/image_raw/compressed",
            self.image_callback, 10
        )

        self.create_subscription(
            String, "/human_action",
            self.action_callback, 10
        )

        self.cmd_pub = self.create_publisher(Twist, "/cmd_vel", 10)

        self.model = YOLO("yolo11n.pt")

        self.get_logger().info("FollowerFSM V1 (FULL DEBUG) started")

    # ==================================================
    # FSM state change logging
    # ==================================================
    def set_state(self, new_state: str):
        if new_state != self.state:
            self.get_logger().warn(f"[FSM] {self.state}  -->  {new_state}")
            self.state = new_state

    def scan_callback(self, msg: LaserScan):
        self.scan = msg

    # ==================================================
    # Action input
    # ==================================================
    def action_callback(self, msg: String):
        p, g = decode_action(msg.data)
        if p is None:
            self.get_logger().warn(f"[ACTION] Invalid format: {msg.data}")
            return

        self.prev_posture = self.posture
        self.posture = p
        self.gesture = g

        self.get_logger().info(f"[ACTION] posture={p}, gesture={g}")

        # Wave toggle service
        if g == "wave":
            if not self.service:
                self.service = True
                self.set_state("FOLLOW")
                self.get_logger().warn("WAVE: Service START")
            return

        if not self.service:
            return

        if g == "reach" and p in ("stand", "sit", "squat"):
            self.set_state("APPROACH_30")
            return

        if self.prev_posture in ("sit", "squat") and p == "stand":
            self.set_state("RETREAT_60")
            return

        if p == "walk":
            self.set_state("FOLLOW")
            return

        if p == "stand":
            self.set_state("APPROACH_60")
            return

        if p in ("sit", "squat"):
            self.set_state("APPROACH_30")
            return

    # ==================================================
    # Image callback
    # ==================================================
    def image_callback(self, msg: CompressedImage):

        arr = np.frombuffer(msg.data, np.uint8)
        img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        if img is None:
            return

        res = self.model(img, classes=[0], verbose=False)

        best_cx = None
        best_bw = 0.0
        max_area = 0.0

        for b in res[0].boxes:
            x1, y1, x2, y2 = b.xyxy[0].tolist()
            bw = x2 - x1
            bh = y2 - y1
            area = bw * bh
            if area > max_area:
                max_area = area
                best_cx = (x1 + x2) / 2.0
                best_bw = bw

        if best_cx is None:
            self.lost_count += 1

            if self.lost_count < self.lost_threshold:
                self.stop()
                return

            self.follow_lidar()
            return
        else:
            self.lost_count = 0

        drift = best_cx - self.W / 2.0

        if self.debug:
            self.get_logger().info(
                f"[DETECT] area={max_area:.1f} bw={best_bw:.1f} drift={drift:.1f}"
            )

        # Approaching detection
        approaching = False
        if self.prev_area is not None and (max_area - self.prev_area) > 8000:
            approaching = True
            if self.debug:
                self.get_logger().info("[EVENT] Human approaching")
        self.prev_area = max_area

        cmd = Twist()

        # Emergency stop
        if best_bw > self.bw_stop:
            if self.debug:
                self.get_logger().warn("[SAFETY] Too close -> STOP")
            self.stop()
            return

        # Always face human
        if self.service or (self.state == "IDLE" and approaching):
            cmd.angular.z = self.turn_to_face(drift)

        # FSM Linear Control
        if self.state == "IDLE":
            cmd.linear.x = 0.0

        elif self.state == "FOLLOW":
            if best_bw < self.bw_60:
                cmd.linear.x = 0.15
            else:
                cmd.linear.x = 0.0

        elif self.state == "APPROACH_60":
            cmd.linear.x = self.dist_ctrl(best_bw, self.bw_60)

        elif self.state == "APPROACH_30":
            cmd.linear.x = self.dist_ctrl(best_bw, self.bw_30)
            if best_bw >= self.bw_30:
                cmd.linear.x = 0.0

        elif self.state == "RETREAT_60":
            if best_bw > self.bw_60:
                cmd.linear.x = -0.12
            else:
                cmd.linear.x = 0.0
                if self.posture == "walk":
                    self.set_state("FOLLOW")
                else:
                    self.set_state("APPROACH_60")

        if self.debug:
            self.get_logger().info(
                f"[CMD] state={self.state} "
                f"linear={cmd.linear.x:.2f} "
                f"angular={cmd.angular.z:.2f}"
            )

        self.cmd_pub.publish(cmd)


    def follow_lidar(self):
        if self.scan is None:
            self.get_logger().warn("[LIDAR] No scan data")
            self.stop()
            return

        ranges = np.array(self.scan.ranges)

        # 去掉无效值
        ranges = np.where(np.isfinite(ranges), ranges, 10.0)

        # 找最近物体
        min_dist = np.min(ranges)
        min_idx = np.argmin(ranges)

        angle = self.scan.angle_min + min_idx * self.scan.angle_increment

        cmd = Twist()

        # 👉 只转向，不前进（你要求的是 turn to it）
        cmd.angular.z = -0.8 * angle

        if self.debug:
            self.get_logger().info(
                f"[LIDAR] nearest={min_dist:.2f}m angle={angle:.2f} -> turn={cmd.angular.z:.2f}"
            )

        self.cmd_pub.publish(cmd)
    # ==================================================
    # Turn control
    # ==================================================
    def turn_to_face(self, drift: float) -> float:
        if abs(drift) < self.deadband:
            if self.debug:
                self.get_logger().info("[TURN] Within deadband")
            return 0.0

        ang = -self.k_ang * drift
        if self.debug:
            self.get_logger().info(
                f"[TURN] drift={drift:.1f} -> angular={ang:.3f}"
            )
        return ang

    # ==================================================
    # Distance control
    # ==================================================
    def dist_ctrl(self, bw: float, target_bw: float) -> float:
        if bw < target_bw - 15:
            if self.debug:
                self.get_logger().info("[DIST] Too far -> forward")
            return 0.12

        if bw > target_bw + 25:
            if self.debug:
                self.get_logger().info("[DIST] Too close -> backward")
            return -0.10

        if self.debug:
            self.get_logger().info("[DIST] In range -> stop")
        return 0.0

    # ==================================================
    # Stop
    # ==================================================
    def stop(self):
        self.cmd_pub.publish(Twist())


def main():
    rclpy.init()
    node = FollowerFSMV1()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()