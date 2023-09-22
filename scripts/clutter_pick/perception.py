
from typing import Dict

from ultralytics import YOLO
import numpy as np
import cv2


from sensor_msgs.msg import Image
import cv_bridge


class YoloDetector:
    '''
    YoloDetector class

    Attributes:
        model (str): path to model
    '''
    def __init__(self, model='yolov8n.pt'):

        self.model = YOLO(model)
        self.bridge = cv_bridge.CvBridge()

    def detect(self, img:Image)->(np.array, Dict[str, np.array]):
        '''
        Detect objects in an image
        
        Args:
            img (Image): image to detect objects in

        Returns:
            res_plotted (np.array): image with detected objects
            predicted_boxes (np.array): array of bounding boxes
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
        predicted_classes = results[0].boxes.cls

        masks = {}

        for i, mask in enumerate(predicted_masks):
            # check if mask is empty
            if np.count_nonzero(mask) == 0:
                continue
            masks[self.model.names[int(predicted_classes[i])]] = mask

        return res_plotted, masks
    


