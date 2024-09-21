from launch import LaunchDescription
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory

def generate_launch_description():
    return LaunchDescription([
        Node(
            package='car_detector',
            executable='camera_publisher',
            name='camera_publisher',
            output='screen'
        ),
        Node(
            package='car_detector',
            executable='camera_subscriber',
            name='camera_subscriber',
            output='screen'
        ),
        Node(
            package='rviz2',
            executable='rviz2',
            name='rviz2',
            output='screen',
            arguments=['-d', [get_package_share_directory('car_detector'), '/rviz/camera_config.rviz']]
        )
    ])

