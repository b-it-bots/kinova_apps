from typing import List
from ultralytics import YOLO
import numpy as np
import cv2

OBJECTS = {'toothbrush': 79, 'cup': 41,'fork': 42,'knife': 43,'spoon': 44,'bowl': 45}

class YoloDetector:

    def __init__(self, model='yolov8n.pt'):
        self.model = YOLO(model)

    def detect(self, img: np.array):
        results = self.model.predict(source=img, conf=0.25, classes=[OBJECTS['cup'], OBJECTS['toothbrush']])

        predicted_boxes = results[0].boxes.xyxy.cpu().numpy().astype(int)

        return predicted_boxes
    


    