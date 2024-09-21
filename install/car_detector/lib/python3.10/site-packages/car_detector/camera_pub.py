from __future__ import print_function
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2

class CameraPublisher(Node):
    """
    This class publishes camera frames captured from a camera to a ROS topic.
    """
    def __init__(self):
        super().__init__('camera_publisher')
        self.publisher_ = self.create_publisher(Image, 'camera_frames', 10)
        self.br = CvBridge()
        self.timer_period = 0.1  # seconds (10 Hz)

        # Initialize the video capture from the camera.
        # Change 0 to 1 or 2 if you have multiple cameras and the default isn't the desired one.
        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            self.get_logger().error('Could not open camera device')
            rclpy.shutdown()

        # Create the timer.
        self.timer = self.create_timer(self.timer_period, self.timer_callback)

    def timer_callback(self):
        ret, frame = self.cap.read()
        if ret:
            # Publish the current frame
            self.publisher_.publish(self.br.cv2_to_imgmsg(frame, 'bgr8'))
            self.get_logger().info('Publishing camera frame')
        else:
            self.get_logger().warn('Frame capture failed')

    def destroy_node(self):
        # Release the camera capture on node destruction
        self.cap.release()
        super().destroy_node()

def main(args=None):
    rclpy.init(args=args)
    camera_publisher = CameraPublisher()
    try:
        rclpy.spin(camera_publisher)
    except KeyboardInterrupt:
        pass
    finally:
        # Destroy the node explicitly
        camera_publisher.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()