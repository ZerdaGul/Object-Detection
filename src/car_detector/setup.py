from setuptools import setup
from glob import glob
import os

package_name = 'car_detector'

setup(
    name=package_name,
    version='0.0.0',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        ('share/' + package_name + '/launch', glob('launch/*.py')),
        ('share/' + package_name + '/config', glob('config/*.rviz')),
        ('share/' + package_name + '/data', glob('data/*.pt')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='robot',
    maintainer_email='todo@todo.todo',
    description='Car detection package using YOLO model',
    license='TODO: License declaration',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'camera_publisher = car_detector.camera_pub:main',
            'car_detection = car_detector.car_detection:main',
        ],
    },
)
