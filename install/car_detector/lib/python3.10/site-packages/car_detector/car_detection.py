from __future__ import print_function
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2
from ultralytics import YOLO
from ament_index_python.packages import get_package_share_directory
import numpy as np  # numpy modülünü ekleyin

class ImageProcessor(Node):
    def __init__(self):
        super().__init__('image_processor')
        package_share_directory = get_package_share_directory('car_detector')
        self.br = CvBridge()
        
        # 'data' klasöründeki model dosyasının tam yolunu alın
        path = "/data/best.pt"
        full_path = package_share_directory + path
        self.model = YOLO(full_path)
        self.f_path= full_path
        
        self.subscription = self.create_subscription(
            Image, 'camera_frames', self.listener_callback, 10)
        
        self.publisher_ = self.create_publisher(Image, 'processed_frames', 10)

    def listener_callback(self, data):
        self.get_logger().info('Receiving camera frame')
        self.get_logger().info('******************PACKAGE_SHARE_DIR: '+ self.f_path)
        
        current_frame = self.br.imgmsg_to_cv2(data)
        
        results = self.model.predict(source=current_frame, conf=0.8)
        
        # Process and draw results on the frame
        for result in results:
            for box in result.boxes:
                xyxy = box.xyxy[0]  # İlk sonucu alıyoruz
                x1, y1, x2, y2 = map(int, xyxy)  # Dört köşeyi int'e çeviriyoruz
                cv2.rectangle(current_frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
        
        self.publisher_.publish(self.br.cv2_to_imgmsg(current_frame, 'bgr8'))
        cv2.imshow("Processed Frame", current_frame)
        cv2.waitKey(1)

def main(args=None):
    rclpy.init(args=args)
    image_processor = ImageProcessor()
    try:
        rclpy.spin(image_processor)
    except KeyboardInterrupt:
        pass
    finally:
        image_processor.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
