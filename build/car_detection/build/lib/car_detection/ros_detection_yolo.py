#!/usr/bin/env python3
import os, sys
import rclpy
from rclpy.node import Node

from sensor_msgs.msg import Image, CompressedImage
from cv_bridge import CvBridge, CvBridgeError
import cv2

#import rospkg

import numpy as np
import time
from pathlib import Path
import torch
import torch.backends.cudnn as cudnn

#bunları silebilirsin belki
import argparse
import csv
import platform
# buraya kadarını

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # YOLOv5 root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative

#eklediklerim
from yolov5.models.experimental import attempt_load
from yolov5.utils.augmentations import Albumentations, augment_hsv, copy_paste, letterbox, mixup, random_perspective
from yolov5.utils.plots import colors #burda plot_one_box vardı onun yerine Annotator.box_label denicem!!!
#load_classifier yerine reshape_classifier_output denicem!!!
#time_synchronized yerine time_sync denicem!!!
from yolov5.utils.torch_utils import select_device, reshape_classifier_output, time_sync

#buraya kadar


from ultralytics.utils.plotting import Annotator, colors, save_one_box

from yolov5.models.common import DetectMultiBackend
from yolov5.utils.dataloaders import IMG_FORMATS, VID_FORMATS, LoadImages, LoadScreenshots, LoadStreams
from yolov5.utils.general import (
    LOGGER,
    Profile,
    check_file,
    check_img_size,
    check_imshow,
    check_requirements,
    colorstr,
    cv2,
    increment_path,
    non_max_suppression,
    print_args,
    scale_boxes,
    strip_optimizer,
    xyxy2xywh,
)
from yolov5.utils.torch_utils import select_device, smart_inference_mode

bridge= CvBridge()

#rospack = rospkg.RosPack()
#package_path = rospack.get_path('car_detection')

class Camera_subscriber(Node):
    def __init__(self):
        super().__init__('camera_subscriber')
        self.source= '0'
        #self.weights = os.path.join(package_path, 'yolov5', 'runs', 'train', 'exp', 'weights', 'best.pt')
        self.weights= 'src/car_detection/yolov5/runs/train/exp/weights/best.pt'
        self.imgsz = (640, 640)
        self.conf_thres=0.25
        self.iou_thres=0.45
        self.max_det=1000
        self.classes=None
        self.agnostic_nms=False
        self.augment=False
        self.visualize=True
        self.line_thickness=3 
        self.hide_labels=False
        self.hide_conf=False
        self.half=False
        self.save_txt=True
        self.save_csv=True
        self.stride = 16
        self.vid_stride=0
        device_num=''
        view_img=True
        save_crop=False
        nosave=False 
        update=False
        name='exp'
        self.dnn= False 
        #self.data= os.path.join(package_path, 'yolov5', 'data', 'coco128.yaml') 
        self.data= 'src/car_detection/yolov5/data/coco128.yaml'
        self.half=False
        #self.project= os.path.join(package_path, 'yolov5', 'runs', 'detect') 
        self.project= 'src/car_detection/yolov5/runs/detect'
        self.exist_ok= False
        self.device= select_device(device_num)
        self.half &= self.device.type != 'cpu'
        self.save_txt=False

        self.save_dir = increment_path(Path(self.project) / name, exist_ok=self.exist_ok)  # increment run
        (self.save_dir / "labels" if self.save_txt else self.save_dir).mkdir(parents=True, exist_ok=True)  # make dir


        #Load Model
        self.model = DetectMultiBackend(weights=self.weights, device=self.device, dnn=self.dnn, data=self.data, fp16=self.half)
        stride, names, pt = self.model.stride, self.model.names, self.model.pt
        imgsz = check_img_size(self.imgsz, s=stride)  # check image size

        #Data Loader
        view_img = check_imshow(warn=True)
        self.dataset = LoadStreams(self.source, img_size=imgsz, stride=stride, auto=pt, vid_stride=self.vid_stride)
        bs = len(self.dataset)

        #Run Inference
        self.model.warmup(imgsz=(1 if pt or self.model.triton else bs, 3, *imgsz))  # warmup
        

        self.subscription = self.create_subscription(
            Image,
            'rgb_cam/image_raw',
            self.camera_callback,
            10)
        self.subscription  # prevent unused variable warning

    def camera_callback(self, data):
        seen, windows, self.dt = 0, [], (Profile(device=self.device), Profile(device=self.device), Profile(device=self.device))
        for path, im, im0s, vid_cap, s in self.dataset:
            with self.dt[0]:
                im = bridge.imgmsg_to_cv2(data, "bgr8")
                # check for common shapes
                s = np.stack([letterbox(x, self.imgsz, stride=self.stride)[0].shape for x in im], 0)  # shapes
                self.rect = np.unique(s, axis=0).shape[0] == 1  # rect inference if all shapes equal
                if not self.rect:
                    print('WARNING: Different stream shapes detected. For optimal performance supply similarly-shaped streams.')

                # Letterbox
                im0 = im.copy()
                im = im[np.newaxis, :, :, :]        

                # Stack
                im = np.stack(im, 0)

                # Convert
                im = im[..., ::-1].transpose((0, 3, 1, 2))  # BGR to RGB, BHWC to BCHW
                im = np.ascontiguousarray(im)




                im = torch.from_numpy(im).to(self.model.device)
                im = im.half() if self.model.fp16 else im.float()  # uint8 to fp16/32
                im /= 255  # 0 - 255 to 0.0 - 1.0
                if len(im.shape) == 3:
                    im = im[None]  # expand for batch dim
                if self.model.xml and im.shape[0] > 1:
                    ims = torch.chunk(im, im.shape[0], 0)

            # Inference
            with self.dt[1]:
                visualize = increment_path(self.save_dir / Path(path).stem, mkdir=True) if self.visualize else False
                if self.model.xml and im.shape[0] > 1:
                    pred = None
                    for image in ims:
                        if pred is None:
                            pred = self.model(image, augment=self.augment, visualize=self.visualize).unsqueeze(0)
                        else:
                            pred = torch.cat((pred, self.model(image, augment=self.augment, visualize=visualize).unsqueeze(0)), dim=0)
                    pred = [pred, None]
                else:
                    pred = self.model(im, augment=self.augment, visualize=self.visualize)
            # NMS
            with self.dt[2]:
                pred = non_max_suppression(pred, self.conf_thres, self.iou_thres, self.classes, self.agnostic_nms, max_det=self.max_det)

        # Second-stage classifier (optional)
        # pred = utils.general.apply_classifier(pred, classifier_model, im, im0s)

        # Define the path for the CSV file
        csv_path = self.save_dir / "predictions.csv"

        # Create or append to the CSV file
        def write_to_csv(image_name, prediction, confidence):
            """Writes prediction data for an image to a CSV file, appending if the file exists."""
            data = {"Image Name": image_name, "Prediction": prediction, "Confidence": confidence}
            with open(csv_path, mode="a", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=data.keys())
                if not csv_path.is_file():
                    writer.writeheader()
                writer.writerow(data)

        # Process detections
        for i, det in enumerate(pred):  # detections per image
            seen +=1
            p = path[i]
            im0= im0s[i].copy()
            s = f'{i}: '
            s += '%gx%g ' % im.shape[2:]  # print string
            p = Path(p)
            save_path = str(self.save_dir / p.name)  # im.jpg
            txt_path = str(self.save_dir / "labels" / p.stem) + ("" if self.dataset.mode == "image" else f"_{self.frame}")  # im.txt
            s += "%gx%g " % im.shape[2:]  # print string
            gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
            imc = im0.copy() if self.save_crop else im0  # for save_crop
            annotator = Annotator(im0, line_width=self.line_thickness, example=str(self.names))
            
            if len(det):
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_boxes(im.shape[2:], det[:, :4], im0.shape).round()

                # Print results
                for c in det[:, 5].unique():
                    n = (det[:, 5] == c).sum()  # detections per class
                    s += f"{n} {self.names[int(c)]}{'s' * (n > 1)}, "  # add to string

                for *xyxy, conf, cls in reversed(det):
                    c = int(cls)  # integer class
                    label = self.names[c] if self.hide_conf else f"{self.names[c]}"
                    confidence = float(conf)
                    confidence_str = f"{confidence:.2f}"

                    if self.save_csv:
                        write_to_csv(p.name, label, confidence_str)

                    if self.save_txt:  # Write to file
                        xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh
                        line = (cls, *xywh, conf) if self.save_conf else (cls, *xywh)  # label format
                        with open(f"{txt_path}.txt", "a") as f:
                            f.write(("%g " * len(line)).rstrip() % line + "\n")

                    if self.save_img or self.save_crop or self.view_img:  # Add bbox to image
                        c = int(cls)  # integer class
                        label = None if self.hide_labels else (self.names[c] if self.hide_conf else f"{self.names[c]} {conf:.2f}")
                        annotator.box_label(xyxy, label, color=colors(c, True))
                    if self.save_crop:
                        save_one_box(xyxy, imc, file=self.save_dir / "crops" / self.names[c] / f"{p.stem}.jpg", BGR=True)

                im0 = annotator.result()
                if self.view_img:
                    if platform.system() == "Linux" and p not in windows:
                        windows.append(p)
                        cv2.namedWindow(str(p), cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)  # allow window resize (Linux)
                        cv2.resizeWindow(str(p), im0.shape[1], im0.shape[0])
                    cv2.imshow(str(p), im0)
                    cv2.waitKey(1)  # 1 millisecond

                # Save results (image with detections)
                if self.save_img:
                    if self.dataset.mode == "image":
                        cv2.imwrite(save_path, im0)
                    else:  # 'video' or 'stream'
                        if self.vid_path[i] != save_path:  # new video
                            self.vid_path[i] = save_path
                            if isinstance(self.vid_writer[i], cv2.VideoWriter):
                                self.vid_writer[i].release()  # release previous video writer
                            if vid_cap:  # video
                                fps = vid_cap.get(cv2.CAP_PROP_FPS)
                                w = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                                h = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                            else:  # stream
                                fps, w, h = 30, im0.shape[1], im0.shape[0]
                            save_path = str(Path(save_path).with_suffix(".mp4"))  # force *.mp4 suffix on results videos
                            self.vid_writer[i] = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
                        self.vid_writer[i].write(im0)

            # Print time (inference-only)
            LOGGER.info(f"{s}{'' if len(det) else '(no detections), '}{self.dt[1].dt * 1E3:.1f}ms")

            # Print results
            t = tuple(x.t / seen * 1e3 for x in self.dt)  # speeds per image
            LOGGER.info(f"Speed: %.1fms pre-process, %.1fms inference, %.1fms NMS per image at shape {(1, 3, *(self.imgsz))}" % t)
            if self.save_txt or self.save_img:
                s = f"\n{len(list(self.save_dir.glob('labels/*.txt')))} labels saved to {self.save_dir / 'labels'}" if self.save_txt else ""
                LOGGER.info(f"Results saved to {colorstr('bold', self.save_dir)}{s}")
            if self.update:
                strip_optimizer(self.weights)  # update model (to fix SourceChangeWarning)


def main():
    rclpy.init(args=None)
    camera_subscriber = Camera_subscriber()
    rclpy.spin(camera_subscriber)
    rclpy.shutdown()


if __name__ == '__main__':
    main()

