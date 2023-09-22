from typing import List
from ultralytics import YOLO
from ultralytics.utils.plotting import Annotator 
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
        img = self.bridge.imgmsg_to_cv2(img, desired_encoding='passthrough')
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = self.model.predict(source=img, conf=0.25)

        for r in results:
        
            annotator = Annotator(img)
            
            boxes = r.boxes
            for box in boxes:
                
                b = box.xyxy[0]  # get box coordinates in (top, left, bottom, right) format
                c = box.cls
                annotator.box_label(b, self.model.names[int(c)])
            
        frame = annotator.result()

        predicted_boxes = results[0].boxes.xyxy.cpu().numpy().astype(int)

        return frame, predicted_boxes
    


    