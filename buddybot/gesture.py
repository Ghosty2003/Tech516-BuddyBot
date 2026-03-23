import rclpy
from rclpy.node import Node
from std_msgs.msg import String

import cv2
import mediapipe as mp
import numpy as np
from cv_bridge import CvBridge
from sensor_msgs.msg import CompressedImage


# -----------------------------
# Angle Calculation
# -----------------------------
def calculate_angle(a, b, c):
    a = np.array([a.x, a.y])
    b = np.array([b.x, b.y])
    c = np.array([c.x, c.y])

    ba = a - b
    bc = c - b

    cos_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-6)
    cos_angle = np.clip(cos_angle, -1.0, 1.0)
    return np.degrees(np.arccos(cos_angle))


# -----------------------------
# ROS2 Node
# -----------------------------
class HumanActionPublisher(Node):

    def __init__(self):
        super().__init__('human_action_publisher')

        # Action Publisher
        self.pub = self.create_publisher(String, '/human_action', 10)

        # Camera Subscription
        self.sub = self.create_subscription(
            CompressedImage,
            '/image_raw/compressed',
            self.image_callback,
            10
        )

        self.bridge = CvBridge()

        # MediaPipe
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(static_image_mode=False)

        self.get_logger().info("Human Action Publisher Started")

        # -----------------------------
        # Stability Control (Debouncing)
        # -----------------------------
        self.last_detected = None
        self.detect_count = 0

        # -----------------------------
        # Publishing Control (Cooldown)
        # -----------------------------
        self.last_published_pose = None
        self.last_publish_time = self.get_clock().now()

        # Frame rate control
        self.last_time = self.get_clock().now()

    # -----------------------------
    # Image Callback
    # -----------------------------
    def image_callback(self, msg: CompressedImage):

        now = self.get_clock().now()

        # Limit Frequency (5Hz)
        if (now - self.last_time).nanoseconds < 0.2e9:
            return
        self.last_time = now

        # -----------------------------
        # Decode Image
        # -----------------------------
        np_arr = np.frombuffer(msg.data, np.uint8)
        frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

        if frame is None:
            self.get_logger().warn("Failed to decode image")
            return

        cv2.imshow("camera", frame)
        cv2.waitKey(1)

        # -----------------------------
        # MediaPipe Processing
        # -----------------------------
        image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.pose.process(image_rgb)

        pose_label = "No Person"

        if results.pose_landmarks:
            lm = results.pose_landmarks.landmark

            right_shoulder = lm[self.mp_pose.PoseLandmark.RIGHT_SHOULDER]
            right_elbow = lm[self.mp_pose.PoseLandmark.RIGHT_ELBOW]
            right_wrist = lm[self.mp_pose.PoseLandmark.RIGHT_WRIST]

            left_hip = lm[self.mp_pose.PoseLandmark.LEFT_HIP]
            right_hip = lm[self.mp_pose.PoseLandmark.RIGHT_HIP]
            left_knee = lm[self.mp_pose.PoseLandmark.LEFT_KNEE]
            right_knee = lm[self.mp_pose.PoseLandmark.RIGHT_KNEE]
            left_ankle = lm[self.mp_pose.PoseLandmark.LEFT_ANKLE]
            right_ankle = lm[self.mp_pose.PoseLandmark.RIGHT_ANKLE]

            # Angle Calculation
            left_leg_angle = calculate_angle(left_hip, left_knee, left_ankle)
            right_leg_angle = calculate_angle(right_hip, right_knee, right_ankle)
            arm_angle = calculate_angle(right_shoulder, right_elbow, right_wrist)

            # -----------------------------
            # Action Recognition Logic
            # -----------------------------
            if right_wrist.y < right_shoulder.y:
                pose_label = "Wave"
            elif arm_angle < 45:
                pose_label = "Reach Out"
            elif left_leg_angle < 120 and right_leg_angle < 120:
                pose_label = "Sitting"
            elif left_leg_angle > 160 and right_leg_angle > 160:
                pose_label = "Standing"

        # -----------------------------
        # consecutive detections (Debouncing)
        # -----------------------------
        if pose_label == self.last_detected:
            self.detect_count += 1
        else:
            self.last_detected = pose_label
            self.detect_count = 1

        if self.detect_count < 3:
            return

        # -----------------------------
        # ✅ Cooldown mechanism (Sit / Stand 3 seconds)
        # -----------------------------
        if pose_label in ["Sitting", "Standing"]:
            dt = (now - self.last_publish_time).nanoseconds / 1e9
            if pose_label == self.last_published_pose and dt < 3.0:
                self.get_logger().info(f"{pose_label} cooling down...")
                return

        # -----------------------------
        # Publish Action
        # -----------------------------
        out_msg = String()
        out_msg.data = pose_label
        self.pub.publish(out_msg)

        self.last_published_pose = pose_label
        self.last_publish_time = now

        self.get_logger().info(f"Published Action: {pose_label}")


# -----------------------------
# Main
# -----------------------------
def main(args=None):
    rclpy.init(args=args)
    node = HumanActionPublisher()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()