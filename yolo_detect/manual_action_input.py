#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from std_msgs.msg import String


HELP = """
Manual Action Input:
 gesture: a=none, b=wave, c=reach
 posture: 0=walk, 1=stand, 2=sit, 3=squat
Example:
 1a  (stand + none)
 0b  (walk + wave)
 2c  (sit  + reach)
Type 'q' to quit.
"""


class ManualActionInput(Node):
   def __init__(self):
       super().__init__("manual_action_input")
       self.pub = self.create_publisher(String, "/human_action", 10)
       self.get_logger().info("ManualActionInput started. Publishing to /human_action")
       self.get_logger().info(HELP)


   def run(self):
       while rclpy.ok():
           s = input("Enter action (e.g., 1a,2c,0b) > ").strip().lower()
           if s in ("q", "quit", "exit"):
               break


           if len(s) != 2 or s[0] not in "0123" or s[1] not in "abc":
               print("Invalid. Use like 1a / 2c / 0b. See help above.")
               continue


           msg = String()
           msg.data = s
           self.pub.publish(msg)
           self.get_logger().info(f"Published /human_action: {s}")


def main():
   rclpy.init()
   node = ManualActionInput()
   try:
       node.run()
   finally:
       node.destroy_node()
       rclpy.shutdown()


if __name__ == "__main__":
   main()
