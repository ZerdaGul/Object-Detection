from launch import LaunchDescription
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory
import os

def generate_launch_description():
    package_share_directory = get_package_share_directory('car_detector')
    rviz_config_file = os.path.join(package_share_directory, 'config', 'camera_car_detection.rviz')

    return LaunchDescription([
        Node(
            package='car_detector',
            executable='camera_publisher',
            name='camera_publisher',
            output='screen'
        ),
        Node(
            package='car_detector',
            executable='car_detection',
            name='car_detection',
            output='screen'
        ),
        Node(
            package='rviz2',
            executable='rviz2',
            name='rviz2',
            output='screen',
            arguments=['-d', rviz_config_file]
        )
    ])
