import rclpy
from rclpy.node import Node
import numpy as np
import cv2
from sensor_msgs.msg import CompressedImage
from geometry_msgs.msg import Pose2D  # 简单起见，用 Pose2D 发布 x, y, size
from cv_bridge import CvBridge
from ultralytics import YOLO

class BiggestPersonTracker(Node):
    def __init__(self):
        super().__init__('biggest_person_tracker')
        
        # 订阅原始图像
        self.subscription = self.create_subscription(
            CompressedImage, '/image_raw/compressed', self.image_callback, 10)
        
        # 发布最大目标的坐标 (x, y 为中心点, theta 为面积)
        self.publisher_ = self.create_publisher(Pose2D, '/target_person_pos', 10)
        
        self.bridge = CvBridge()
        self.model = YOLO("yolo11n.pt")
        self.get_logger().info('Tracker Started. Publishing biggest person to /target_person_pos')

    def image_callback(self, msg):
        try:
            # 解码压缩的图像
            np_arr = np.frombuffer(msg.data, np.uint8)
            cv_image = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
            
            # 如果解码成功，继续进行目标检测
            results = self.model(cv_image, classes=[0], verbose=False)

            biggest_box = None
            max_area = 0

            # 遍历所有检测到的人
            for box in results[0].boxes:
                coords = box.xyxy[0].tolist()
                width = coords[2] - coords[0]
                height = coords[3] - coords[1]
                area = width * height
                
                # 寻找面积最大的框
                if area > max_area:
                    max_area = area
                    center_x = (coords[0] + coords[2]) / 2
                    center_y = (coords[1] + coords[3]) / 2
                    biggest_box = (center_x, center_y, area)

            # 如果找到了人，发布信息
            if biggest_box:
                pos_msg = Pose2D()
                pos_msg.x = float(biggest_box[0])  # 屏幕 X 坐标
                pos_msg.y = float(biggest_box[1])  # 屏幕 Y 坐标
                pos_msg.theta = float(biggest_box[2])  # 面积 (代表远近)
                
                self.publisher_.publish(pos_msg)
                self.get_logger().info(f'Target: x={pos_msg.x:.1f}, area={pos_msg.theta:.0f}')

        except Exception as e:
            self.get_logger().error(f'Error: {e}')


def main():
    rclpy.init()
    node = BiggestPersonTracker()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()