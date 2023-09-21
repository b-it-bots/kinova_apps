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

        four_coordinates = []

        for object in predicted_boxes:
             x1 = object[0]
             y1 = object[1]
             x2 = object[2]
             y2 = object[3]
             four_coordinates.append([(x1,y1), (x2,y1), (x1,y2), (x2,y2)])
        
        return four_coordinates
    


    