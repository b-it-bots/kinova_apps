from typing import List

from ultralytics.utils.plotting import Annotator  
from ultralytics import YOLO
import numpy as np
import cv2

OBJECTS = {'toothbrush': 79, 'cup': 41,'fork': 42,'knife': 43,'spoon': 44,'bowl': 45}

class YoloDetector:

    def __init__(self, model='yolov8n.pt'):
        self.model = YOLO(model)

    def detect(self, img: np.array):
        results = self.model.predict(source=img, conf=0.25, classes=[OBJECTS['cup'], OBJECTS['toothbrush'], OBJECTS['fork'], OBJECTS['knife'], OBJECTS['spoon']])

        for r in results:
        
            annotator = Annotator(frame)
            
            boxes = r.boxes
            for box in boxes:
                
                b = box.xyxy[0]  # get box coordinates in (top, left, bottom, right) format
                c = box.cls
                annotator.box_label(b, self.model.names[int(c)])
            
        frame = annotator.result()

        predicted_boxes = results[0].boxes.xyxy.cpu().numpy().astype(int)

        return frame, predicted_boxes
    


    