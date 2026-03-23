import os
from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description():
    package_name = 'buddybot'

    yolo_node = Node(
        package=package_name,
        executable='yolo',
        name='yolo_detector',
        output='screen'
    )

    follow_face_node = Node(
        package=package_name,
        executable='follow_face',
        name='follow_face_logic',
        output='screen'
    )

    platform_node = Node(
        package=package_name,
        executable='platform',
        name='platform_controller',
        output='screen'
    )

    return LaunchDescription([
        yolo_node,
        follow_face_node,
        platform_node,

    ])