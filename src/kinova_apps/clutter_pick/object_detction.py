from typing import List
from ultralytics import YOLO
import numpy as np
import cv2
import cv_bridge
from sensor_msgs.msg import Image

OBJECTS = {'toothbrush': 79, 'cup': 41,'fork': 42,'knife': 43,'spoon': 44,'bowl': 45}

class YoloDetector:

    def __init__(self, model='yolov8n.pt'):
        self.model = YOLO(model)
        self.bridge = cv_bridge.CvBridge()

    def detect(self, img: Image):
        # convert image to cv2
        cv_image = self.bridge.imgmsg_to_cv2(img, desired_encoding='passthrough')

        # convert to bgr
        cv_image = cv2.cvtColor(cv_image, cv2.COLOR_RGB2BGR)

        results = self.model.predict(source=cv_image, conf=0.25)

        predicted_boxes = results[0].boxes.xyxy.cpu().numpy().astype(int)

        return predicted_boxes
    


    