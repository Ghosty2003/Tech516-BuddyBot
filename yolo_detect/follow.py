import rclpy
from rclpy.node import Node
from sensor_msgs.msg import CompressedImage
from geometry_msgs.msg import Pose2D, Twist
from std_msgs.msg import String  # For stop command
from cv_bridge import CvBridge
import numpy as np
import cv2
from ultralytics import YOLO

class BiggestPersonTracker(Node):
    def __init__(self):
        super().__init__('biggest_person_tracker')

        # 订阅压缩图像数据
        self.subscription = self.create_subscription(
            CompressedImage, '/image_raw/compressed', self.image_callback, 10)
        
        # 发布目标人物的坐标 (x, y 为中心点, theta 为面积)
        self.publisher_ = self.create_publisher(Pose2D, '/target_person_pos', 10)
        
        # 发布机器人速度 (控制运动)
        self.velocity_publisher = self.create_publisher(Twist, '/cmd_vel', 10)
        
        # 订阅停止命令
        self.stop_subscription = self.create_subscription(
            String, '/stop_robot', self.stop_callback, 10)

        # 初始化组件
        self.bridge = CvBridge()
        self.model = YOLO("yolo11n.pt")
        self.get_logger().info('Tracker Started. Publishing biggest person to /target_person_pos')

        # 停止标志
        self.stop_flag = False

        # 图像宽高（根据你的摄像头进行调整）
        self.image_width = 640  # 示例：640 像素
        self.image_height = 480  # 示例：480 像素

    def stop_callback(self, msg):
        """当接收到 'stop' 命令时停止机器人。"""
        if msg.data == "stop":
            self.stop_flag = True
            self.get_logger().info("收到停止命令，机器人将停止。")
        else:
            self.stop_flag = False

    def image_callback(self, msg):
        """图像处理与机器人运动控制的回调函数。"""
        try:
            if self.stop_flag:
                # 如果接收到停止命令，立即停止机器人
                self.stop_robot()
                return

            # 解码压缩图像
            np_arr = np.frombuffer(msg.data, np.uint8)
            cv_image = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

            # 使用 YOLO 检测人
            results = self.model(cv_image, classes=[0], verbose=False)  # 只检测人
            
            biggest_box = None
            max_area = 0

            # 遍历所有检测到的人
            for box in results[0].boxes:
                # 获取坐标: x1, y1, x2, y2
                coords = box.xyxy[0].tolist()
                width = coords[2] - coords[0]
                height = coords[3] - coords[1]
                area = width * height
                
                # 寻找面积最大的框
                if area > max_area:
                    max_area = area
                    # 计算中心点
                    center_x = (coords[0] + coords[2]) / 2
                    center_y = (coords[1] + coords[3]) / 2
                    biggest_box = (center_x, center_y, area)

            if biggest_box:
                pos_msg = Pose2D()
                pos_msg.x = float(biggest_box[0])  # 中心 X 坐标
                pos_msg.y = float(biggest_box[1])  # 中心 Y 坐标
                pos_msg.theta = float(biggest_box[2])  # 目标框的面积
                
                self.publisher_.publish(pos_msg)
                self.get_logger().info(f'Target: x={pos_msg.x:.1f}, area={pos_msg.theta:.0f}')
                
                # 计算与图像中心的偏差
                drift_x = biggest_box[0] - self.image_width / 2
                drift_y = biggest_box[1] - self.image_height / 2

                self.get_logger().info(f"偏差距离: {drift_x:.1f} 像素")

                # 根据偏差调整机器人的运动
                cmd_vel = Twist()

                # 1. 处理转向 (使人保持在图像中心)
                if abs(drift_x) > 40:  # 给中心留一点死区，防止机器人来回抖动
                    cmd_vel.angular.z = -0.002 * drift_x  # 比例系数根据实际灵敏度调整
                
                # 2. 处理前进/停止 (基于检测框的大小)
                if width > 300 or height > 300:
                    # 离得太近了，停止
                    cmd_vel.linear.x = 0.0
                    cmd_vel.angular.z = 0.0  # 停止旋转，确保安全
                    self.get_logger().warn("距离太近！紧急停止。")
                    self.get_logger().warn("Width: {:.1f}, Height: {:.1f}".format(width, height))  # 格式化并记录宽度和高度
                else:
                    # 目标距离较远，前进
                    cmd_vel.linear.x = 0.15  # 慢速前进，避免突然移动
                    self.get_logger().info("目标远。正在前进...")
                    self.get_logger().warn("Width: {:.1f}, Height: {:.1f}".format(width, height))  # 格式化并记录宽度和高度

                # 3. 发布控制指令
                self.velocity_publisher.publish(cmd_vel)

        except Exception as e:
            self.get_logger().error(f'错误: {e}')

    def stop_robot(self):
        """停止机器人。"""
        cmd_vel = Twist()
        cmd_vel.linear.x = 0.0  # 停止前进/后退
        cmd_vel.angular.z = 0.0  # 停止旋转
        self.velocity_publisher.publish(cmd_vel)
        self.get_logger().info("机器人已停止。")

def main():
    rclpy.init()
    node = BiggestPersonTracker()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()