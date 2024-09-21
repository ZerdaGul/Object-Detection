#!/usr/bin/env python3

import torch
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import os
import cv2

from yolov8_msgs.msg import InferenceResult
from yolov8_msgs.msg import Yolov8Inference

class YOLOv8Node(Node):
    def __init__(self):
        super().__init__('yolov8_node')
        model_path = os.path.expanduser('~/ros2_ws/src/yolobot_recognition/runs/detect/train/weights/best.pt')
        self.model = torch.load(model_path)
        self.model.eval()

        self.bridge = CvBridge()
        self.image_sub = self.create_subscription(Image, 'rgb_cam/image_raw', self.image_callback, 10)
        self.inference_pub = self.create_publisher(Yolov8Inference, "/Yolov8_Inference", 1)
        self.result_img_pub = self.create_publisher(Image, "/inference_result", 1)

    def image_callback(self, msg):
        cv_image = self.bridge.imgmsg_to_cv2(msg, "bgr8")
        tensor_image = self.preprocess_image(cv_image)
        with torch.no_grad():
            results = self.model(tensor_image)

        inference_msg = Yolov8Inference()
        inference_msg.header = msg.header
        for det in results.xyxy[0]:
            inf_res = InferenceResult()
            inf_res.class_name = str(det[-1].item())
            inf_res.top = int(det[0].item())
            inf_res.left = int(det[1].item())
            inf_res.bottom = int(det[2].item())
            inf_res.right = int(det[3].item())
            inference_msg.yolov8_inference.append(inf_res)
            cv2.rectangle(cv_image, (inf_res.left, inf_res.top), (inf_res.right, inf_res.bottom), (255, 255, 0))

        self.inference_pub.publish(inference_msg)
        self.result_img_pub.publish(self.bridge.cv2_to_imgmsg(cv_image, "bgr8"))

    def preprocess_image(self, img):
	img_resized = cv2.resize(img, (640, 640))  # Eğitim sırasında kullanılan boyuta göre ayarlayın
	img_normalized = img_resized / 255.0  # Genellikle YOLO modelleri için gereklidir
	img_tensor = torch.from_numpy(img_normalized).permute(2, 0, 1).unsqueeze(0).float()  # PyTorch tensor formatına dönüştürme
	return img_tensor

if __name__ == '__main__':
    rclpy.init()
    node = YOLOv8Node()
    rclpy.spin(node)
    rclpy.shutdown()

