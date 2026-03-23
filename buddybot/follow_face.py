#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Version 2 – Visual Search Edition
Logic: If human disappears, spin in place to find them. No Lidar used.
"""

import rclpy
from rclpy.node import Node
import numpy as np
import cv2
from sensor_msgs.msg import CompressedImage
from geometry_msgs.msg import Twist
from std_msgs.msg import String
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


class FollowerFSMV2(Node):

    def __init__(self):
        super().__init__("follower_fsm_v2_spin_search")

        # ---------------- Parameters ----------------
        self.debug = True
        self.W = 800
        self.H = 600

        self.deadband = 40
        self.k_ang = 0.002
        
        self.bw_20 = 550.0
        self.bw_30 = 450.0
        self.bw_60 = 300.0
        self.bw_stop = 600.0

        # ---------------- FSM & Status ----------------
        self.state = "IDLE"
        self.service = False

        self.posture = None
        self.gesture = None
        self.prev_posture = POSTURE_MAP["0"]  # initialize as "walk"
        self.prev_gesture = GESTURE_MAP["a"]  # initialize as "none"

        self.prev_area = None
        self.lost_count = 0
        self.lost_threshold = 5   # Start spinning after 5 consecutive frames without detection
        self.spin_speed = 0.2     # Rotation speed when target is lost (rad/s)

        self.wave_count = 0
        self.achieved = False
        
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

        # Load Model
        self.model = YOLO("yolo11n.pt")

        self.get_logger().info("FollowerFSM V2 (Spin Search) started - Lidar Disabled")

    def set_state(self, new_state: str):
        if new_state != self.state:
            self.get_logger().warn(f"[FSM] {self.state}  -->  {new_state}")
            self.state = new_state

    def action_callback(self, msg: String):
        p, g = decode_action(msg.data)
        if p is None:
            return

        self.prev_posture = self.posture
        self.prev_gesture = self.gesture
        self.posture = p
        self.gesture = g
        
        if (self.prev_posture != p) or (self.prev_gesture != g):
            self.achieved = False

        if g == "wave":
            self.wave_count += 1
            if self.wave_count == 1 and not self.service:
                # First wave -> Start Service
                self.service = True
                self.set_state("FOLLOW")
                self.get_logger().warn("WAVE: Service START")
            elif self.wave_count >= 2 and self.service:
                # Second wave -> Stop Service
                self.service = False
                self.set_state("IDLE")
                self.get_logger().warn("WAVE: Service STOP")
                self.stop()  # Stop robot movement
            return

        if not self.service:
            return  # If service is not active, ignore other actions

        
        # Following logic only executes after service is started
        if g == "reach" and p == "sit":
            self.set_state("APPROACH_20")
            return
        elif g == "reach" and p in ("stand", "walk"):
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
    

    # ----------------------
    # image_callback
    # ----------------------
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

        cmd = Twist()

        # ------------------------------
        # Spin Search Logic: Only when service hasn't achieved target
        # ------------------------------
        if best_cx is None and self.achieved == False:
            self.lost_count += 1
            if self.lost_count >= self.lost_threshold:
                # Rotate to find target
                cmd.angular.z = 0.5  
                if self.debug:
                    self.get_logger().info("[SEARCH] Target lost, spinning to find (before wave)...")
            self.cmd_pub.publish(cmd)
            return

        # Target found, reset lost counter
        self.lost_count = 0

        if best_cx is None:
            drift = 0.0
        else:
            drift = best_cx - self.W / 2.0

        # Safety Stop
        if best_bw > self.bw_stop:
            self.stop()
            return

        # After service starts, always face the human
        if self.service:
            cmd.angular.z = self.turn_to_face(drift)

        # Linear speed control based on FSM state
        if self.state == "IDLE":
            cmd.linear.x = 0.0

        elif self.state == "FOLLOW":
            cmd.linear.x = 0.15 if best_bw < self.bw_60 else 0.0

        elif self.state == "APPROACH_60" and self.achieved == False:
            if best_bw >= self.bw_60:
                cmd.linear.x = 0.0  # Target reached, stop moving
                self.achieved = True
                if self.debug:
                    self.get_logger().info("[APPROACH_60] Target reached, stopping until human action changes")
            else:
                cmd.linear.x = self.dist_ctrl(best_bw, self.bw_60)

        elif self.state == "APPROACH_30" and self.achieved == False:
            if best_bw >= self.bw_30:
                cmd.linear.x = 0.0  # Target reached
                self.achieved = True
                if self.debug:
                    self.get_logger().info("[APPROACH_30] Target reached, stopping until human action changes")
            else:
                cmd.linear.x = self.dist_ctrl(best_bw, self.bw_30)

        elif self.state == "APPROACH_20" and self.achieved == False:
            if best_bw >= self.bw_20:
                cmd.linear.x = 0.0  # Target reached
                self.achieved = True
                if self.debug:
                    self.get_logger().info("[APPROACH_20] Target reached, stopping until human action changes")
            else:
                cmd.linear.x = self.dist_ctrl(best_bw, self.bw_20)

        elif self.state == "RETREAT_60":
            if best_bw > self.bw_60:
                cmd.linear.x = -0.12
            else:
                cmd.linear.x = 0.0
                self.set_state("FOLLOW")
        
        self.cmd_pub.publish(cmd)
        if self.debug:
            self.get_logger().info(f"[STATUS] achieved = {self.achieved}")

    def turn_to_face(self, drift: float) -> float:
        if abs(drift) < self.deadband:
            return 0.0
        return -self.k_ang * drift

    def dist_ctrl(self, bw: float, target_bw: float) -> float:
        if bw < target_bw - 15: return 0.12
        if bw > target_bw + 25: return -0.10
        return 0.0

    def stop(self):
        self.cmd_pub.publish(Twist())
        if self.debug:
            self.get_logger().warn("[STOP] Robot stopped due to safe condition or target too close")


def main():
    rclpy.init()
    node = FollowerFSMV2()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()