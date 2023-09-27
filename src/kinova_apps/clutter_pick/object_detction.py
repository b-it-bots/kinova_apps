from typing import Dict, List
from ultralytics import YOLO
from ultralytics.utils.plotting import Annotator 
import numpy as np
import cv2
import cv_bridge
from sensor_msgs.msg import Image
from shapely.geometry import Polygon
from collections import defaultdict


OBJECTS = {'toothbrush': 79, 'cup': 41,'fork': 42,'knife': 43,'spoon': 44,'bowl': 45}
GRIPPER_LEFT_POINTS = np.array([[140, 720],[224, 625],[296, 625],[300, 720],[132, 720]])
GRIPPER_RIGHT_POINTS = np.array([[996, 720],[996, 720],[996, 720],[996, 720],[996, 629],[1072, 621],[1160, 720],[988, 720]])

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
    
    def detect_segments(self, img:Image)->(np.array, List[str] ,List[np.array], List[Polygon]):
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

        # add gripper masks
        
        cv2.fillPoly(img, pts=[GRIPPER_LEFT_POINTS, GRIPPER_RIGHT_POINTS], color=(0, 255, 0))

        # detect objects
        results = self.model.predict(source=img, conf=0.6, iou=0.45, retina_masks=True)
        
        # plot results
        res_plotted = results[0].plot(boxes=True, labels=True)

        predicted_masks = results[0].masks.data.cpu().numpy()
        predicted_segments = results[0].masks.segments
        predicted_classes = results[0].boxes.cls


        polygons = []
        classes = []
        class_masks = []

        for i, mask in enumerate(predicted_masks):
            # check if mask is 

            classes.append(self.model.names[int(predicted_classes[i])])
            class_masks.append(mask)
            # make mask into polygon
            polygon = Polygon(predicted_segments[i])
            polygons.append(polygon)

        return res_plotted, classes, class_masks, polygons
    
    

""" if __name__ == '__main__':

    # test detector

    img = cv2.imread('/home/bk/competition/white/white022.jpg')
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    detector = YoloDetector(model='/home/bk/competition/kinova_apps/models/clutter_picking/best_all_object.pt')
    res_plotted, names,masks, polygons = detector.detect_segments(img)

    print(names)

    for i, mask in enumerate(masks):
        cv2.imshow(names[i], mask)
        cv2.waitKey(0)
    cv2.imshow('test', res_plotted)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
 """