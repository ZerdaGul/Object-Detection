import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2
import numpy as np
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).parent / "object_detection" / "yolov5"))

# YOLOv5 ile ilgili kütüphaneler
import torch
from models.common import DetectMultiBackend
from utils.general import non_max_suppression, scale_boxes, check_img_size
from utils.augmentations import letterbox
from utils.plots import Annotator, colors

class ObjectDetector(Node):
    def __init__(self):
        super().__init__('object_detector')
        self.subscription = self.create_subscription(
            Image,
            '/camera/image_raw',
            self.image_callback,
            10)
        self.subscription  # prevent unused variable warning
        self.bridge = CvBridge()
        self.model = self.load_model()

    def load_model(self):
        model_path = Path(__file__).parent / 'yolov5x.pt'  # Model dosya yolu
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        model = DetectMultiBackend(model_path, device=device, data=Path(__file__).parent / 'data/coco128.yaml')
        imgsz = check_img_size([640, 640], s=model.stride)  # model stride kontrolü
        model.warmup(imgsz=(1, 3, *imgsz))  # model için warmup işlemi
        return model

    def image_callback(self, msg):
        # ROS mesajını OpenCV formatına dönüştürme
        cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        # YOLOv5 modeli ile görüntüyü işleme
        results = self.process_image(cv_image)
        # Sonuçları görselleştirme
        self.publish_results(cv_image, results)

    def process_image(self, img):
        # Görüntüyü YOLOv5 modeline uygun hale getirme
        img = letterbox(img, new_shape=640, auto=False)[0]
        img = img.transpose((2, 0, 1))  # HWC to CHW
        img = np.ascontiguousarray(img)
        img = torch.from_numpy(img).to(self.model.device)
        img = img.float()  # uint8 to fp32
        img /= 255.0  # 0 - 255 to 0.0 - 1.0
        if len(img.shape) == 3:
            img = img[None]  # expand for batch dim

        # Model ile tahmin yapma
        pred = self.model(img, augment=False, visualize=False)
        pred = non_max_suppression(pred)  # NMS uygulama
        return pred

    def publish_results(self, img, results):
        for det in results:  # detections per image
            if len(det):
                # Görüntü üzerine bounding box ve etiket ekleme
                annotator = Annotator(img, line_width=3, example=str(self.model.names))
                for *xyxy, conf, cls in reversed(det):
                    c = int(cls)  # integer class
                    label = f'{self.model.names[c]} {conf:.2f}'
                    annotator.box_label(xyxy, label, color=colors(c, True))
                cv2.imshow('YOLOv5 Detection', annotator.result())
                cv2.waitKey(1)

def main(args=None):
    rclpy.init(args=args)
    object_detector = ObjectDetector()
    try:
        rclpy.spin(object_detector)
    except KeyboardInterrupt:
        pass  # Handle Ctrl-C gracefully
    finally:
        # Destroy the node explicitly
        object_detector.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()

