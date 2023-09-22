from typing import Dict, List
from ultralytics import YOLO
from ultralytics.utils.plotting import Annotator 
import numpy as np
import cv2
import cv_bridge
from sensor_msgs.msg import Image
from shapely.geometry import Polygon

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
    
    def detect_segments(self, img:Image)->(np.array, Dict[str, np.array], List[Polygon]):
        '''
        Detect objects in an image
        
        Args:
            img (Image): image to detect objects in

        Returns:
            res_plotted (np.array): image with detected objects
            predicted_masks (np.array): array of masks
            predicted_polygons (List[Polygon]): list of polygons
        '''

        # convert image to numpy array
        img = self.bridge.imgmsg_to_cv2(img, desired_encoding='passthrough')
        
        # convert image to RGB
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # detect objects
        results = self.model.predict(source=img, conf=0.6, iou=0.45, retina_masks=True)
        
        # plot results
        res_plotted = results[0].plot(boxes=False)

        predicted_masks = results[0].masks.data.cpu().numpy()
        predicted_segments = results[0].masks.segments
        predicted_classes = results[0].boxes.cls

        masks = {}
        polygons = []

        for i, mask in enumerate(predicted_masks):
            # check if mask is empty
            if np.count_nonzero(mask) == 0:
                continue
            masks[self.model.names[int(predicted_classes[i])]] = mask
            # make mask into polygon
            polygon = Polygon(predicted_segments[i])
            polygons.append(polygon)

        return res_plotted, masks, polygons
    
    