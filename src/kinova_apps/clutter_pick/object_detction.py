from typing import Dict, List
from ultralytics import YOLO
from ultralytics.utils.plotting import Annotator 
import numpy as np
import cv2
import cv_bridge
from sensor_msgs.msg import Image
from shapely.geometry import Polygon
from collections import defaultdict


GRIPPER_LEFT_POINTS = np.array([[0, 614],[180, 514],[216, 502],[224, 682],[180, 720],[36, 720],[0, 694],[0, 626]])
GRIPPER_RIGHT_POINTS = np.array([[1128, 720],[1076, 674],[1072, 498],[1120, 502],[1280, 582],[1280, 702],[1280, 720],[1128, 720]])

class YoloDetector:

    def __init__(self, model='yolov8n.pt', handle_model='best_handle.pt'):
        self.model = YOLO(model)
        # self.og_model = YOLO('yolo8x-seg.pt')
        self.handle_model = YOLO(handle_model)
        self.bridge = cv_bridge.CvBridge()
    

    def detect(self, img: Image):
        img = self.bridge.imgmsg_to_cv2(img, desired_encoding='passthrough')
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = self.model.predict(source=img, conf=0.75)

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

    def get_handle_mask(self, img, mask):
        res = cv2.bitwise_and(img,img,mask = mask)
        results = self.handle_model.predict(source=res, conf=0.6, iou=0.45, retina_masks=True)

        if len(results[0]) == 0:
            print('no handle found')
            return None
        return results[0].masks.data.cpu().numpy()[0]
        

    
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
        
        cv2.fillPoly(img, pts=[GRIPPER_LEFT_POINTS, GRIPPER_RIGHT_POINTS], color=(255, 255, 255))

        # detect objects
        results = self.model.predict(source=img, conf=0.75, iou=0.45, retina_masks=True)

        if len(results[0]) == 0:
            print('no objects found')
            return img, [], [], []

        # plot results
        res_plotted = results[0].plot(boxes=True, labels=True)

        predicted_masks = results[0].masks.data.cpu().numpy()
        predicted_segments = results[0].masks.xy
        predicted_classes = results[0].boxes.cls


        polygons = []
        classes = []
        class_masks = []

        for i, mask in enumerate(predicted_masks):
            # check if mask is 

            if self.model.names[int(predicted_classes[i])] in ['Dustpan', 'Brush']:
                print(f'getting the handle...')
                mask = self.get_handle_mask(img, mask.astype(np.uint8))

                if mask is None:
                    continue

            classes.append(self.model.names[int(predicted_classes[i])])
            class_masks.append(mask)
            # make mask into polygon
            polygon = Polygon(predicted_segments[i])
            polygons.append(polygon)

        return res_plotted, classes, class_masks, polygons
    
    

# if __name__ == '__main__':

#     # test detector

#     img = cv2.imread('/home/bk/competition/dataset/dataset/brush/brush_24.jpg')
#     img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

#     detector = YoloDetector(model='/home/bk/competition/kinova_apps/models/clutter_picking/best_run2.pt')
#     res_plotted, names,masks, polygons = detector.detect_segments(img)

#     print(names)

#     for i, mask in enumerate(masks):
#         cv2.imshow(names[i], mask)
#         cv2.waitKey(0)
#     cv2.imshow('test', res_plotted)
#     cv2.waitKey(0)
#     cv2.destroyAllWindows()