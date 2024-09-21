from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description():
    return LaunchDescription([
        Node(
            package='car_detector',
            executable='camera_publisher',
            name='pub'
        ),
        Node(
            package='car_detector',
            executable='car_detect_subscriber',
            name='sub'
        )

    ])
