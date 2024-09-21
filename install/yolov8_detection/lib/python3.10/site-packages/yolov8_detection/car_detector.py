import os
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from std_msgs.msg import String
from cv_bridge import CvBridge
import cv2
import torch

class CarDetector(Node):
    def __init__(self):
        super().__init__('car_detector')
        self.subscription = self.create_subscription(
            Image,
            '/rgb_cam/image_raw',
            self.listener_callback,
            10)
        self.publisher = self.create_publisher(String, 'detected_cars_topic', 10)
        self.bridge = CvBridge()

        # Model yükleme için dosya yolu
        # Çalışma dizininizin doğru olduğundan emin olun
        package_directory = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        weights_directory = os.path.join(package_directory, 'src', 'yolov8_detection', 'weights')
        model_path = os.path.join(weights_directory, 'best.pt')
        self.model = torch.load(model_path)
        self.model.eval()  # Modeli değerlendirme moduna al

    def listener_callback(self, msg):
        cv_image = self.bridge.imgmsg_to_cv2(msg, 'bgr8')
        results = self.model(cv_image)
        detections = results.pandas().xyxy[0]
        car_count = (detections['class'] == 2).sum()
        self.publisher.publish(String(data=f"Detected {car_count} cars"))

def main(args=None):
    rclpy.init(args=args)
    car_detector = CarDetector()
    rclpy.spin(car_detector)
    car_detector.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
import os
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from std_msgs.msg import String
from cv_bridge import CvBridge
import cv2
import torch

class CarDetector(Node):
    def __init__(self):
        super().__init__('car_detector')
        self.subscription = self.create_subscription(
            Image,
            '/rgb_cam/image_raw',
            self.listener_callback,
            10)
        self.publisher = self.create_publisher(String, 'detected_cars_topic', 10)
        self.bridge = CvBridge()

        # Model yükleme için kesin dosya yolu
        model_path = '/home/zerda/ros2_ws/src/yolov8_detection/weights/best.pt'
        self.model = torch.load(model_path)
        self.model.eval()  # Modeli değerlendirme moduna al

    def listener_callback(self, msg):
        cv_image = self.bridge.imgmsg_to_cv2(msg, 'bgr8')
        results = self.model(cv_image)
        detections = results.pandas().xyxy[0]
        car_count = (detections['class'] == 2).sum()
        self.publisher.publish(String(data=f"Detected {car_count} cars"))

def main(args=None):
    rclpy.init(args=args)
    car_detector = CarDetector()
    rclpy.spin(car_detector)
    car_detector.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
