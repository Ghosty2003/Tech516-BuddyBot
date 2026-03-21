import rclpy
from rclpy.node import Node
from std_msgs.msg import String
import torch
import torch.nn as nn
import numpy as np
import cv2
import mediapipe as mp
import time

# -----------------------------
# 模型定义
# -----------------------------
class MLP(nn.Module):
    def __init__(self, input_size, num_classes):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_size, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, num_classes)
        )

    def forward(self, x):
        return self.net(x)

# -----------------------------
# ROS2 节点
# -----------------------------
class HumanActionPublisher(Node):
    def __init__(self, model_path, label_classes, feature_size):
        super().__init__('human_action_publisher')
        self.pub = self.create_publisher(String, '/human_action', 10)
        self.timer = self.create_timer(0.05, self.timer_callback)  # 20Hz 采集帧

        # 初始化 MediaPipe Pose
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(static_image_mode=False)

        # GPU
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # 加载模型
        self.model = MLP(input_size=feature_size, num_classes=len(label_classes)).to(self.device)
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.model.eval()

        self.label_classes = label_classes
        self.feature_size = feature_size

        # 摄像头
        self.cap = cv2.VideoCapture(0)

        # 上一次发布时间
        self.last_publish_time = 0.0
        self.publish_interval = 1.0  # 每秒发布一次

    def timer_callback(self):
        current_time = time.time()
        if current_time - self.last_publish_time < self.publish_interval:
            return  # 不到 1 秒，不预测

        ret, frame = self.cap.read()
        if not ret:
            return

        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.pose.process(image)

        # 提取特征
        feature = self.extract_features(results)
        if feature is not None:
            x_input = torch.tensor([feature], dtype=torch.float32).to(self.device)
            with torch.no_grad():
                pred = self.model(x_input)
                action_idx = pred.argmax(1).item()
                action_label = self.label_classes[action_idx]

            # 发布动作
            msg = String()
            msg.data = action_label
            self.pub.publish(msg)
            self.get_logger().info(f'Predicted action: {action_label}')

            # 更新时间
            self.last_publish_time = current_time
        else:
            self.get_logger().info('No human detected')

    def extract_features(self, results):
        if results.pose_landmarks:
            lm = results.pose_landmarks.landmark
            coords = []
            for point in lm:
                coords.extend([point.x, point.y])
            return np.array(coords, dtype=np.float32)
        else:
            return None

    def destroy_node(self):
        super().destroy_node()
        self.cap.release()
        cv2.destroyAllWindows()

def main(args=None):
    rclpy.init(args=args)
    model_path = 'pose_model.pth'
    feature_size = 66  # 33 keypoints * 2
    label_classes = ['Wave', 'Reach Out', 'Sitting', 'Standing', 'Reach']
    node = HumanActionPublisher(model_path, label_classes, feature_size)
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()