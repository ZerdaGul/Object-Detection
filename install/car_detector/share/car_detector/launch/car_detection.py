from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description():
    return LaunchDescription([
        Node(
            package='car_detector',
            executable='camera_publisher',
            name='pub',
            output='screen'
        ),
        Node(
            package='car_detector',
            executable='car_detect_subscriber',
            name='sub',
            output='screen'
        ),
        Node(
            package='rviz2',
            executable='rviz2',
            name='rviz2',
            output='screen',
            arguments=['-d', '<package_share_directory>/config/camera_car_detection.rviz']
        )
    ])

