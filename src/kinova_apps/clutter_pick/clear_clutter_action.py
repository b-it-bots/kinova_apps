#!/usr/bin/env python3

from typing import List
import rospy

from sensor_msgs.msg import Image

import kortex_driver.msg

import numpy as np
import cv2
import cv_bridge
import matplotlib.pyplot as plt

from shapely.geometry import Polygon, MultiPolygon
from shapely import affinity, plotting

from kinova_apps.abstract_action import AbstractAction
from kinova_apps.full_arm_movement import FullArmMovement
from utils.transform_utils import TransformUtils


class ClearClutterAction(AbstractAction):
    def __init__(
        self,
        arm: FullArmMovement,
        transform_utils: TransformUtils,
        reference_frame: str = "base_link",
    ) -> None:
        super().__init__(arm, transform_utils)

        self.reference_frame = reference_frame

        self.cartesian_velocity_pub = rospy.Publisher(
            "/my_gen3/in/cartesian_velocity", kortex_driver.msg.TwistCommand, queue_size=1
        )

        self.rgb_topic = 'input_image_topic'
        self.depth_topic = 'input_pc_topic'

        self.bridge = cv_bridge.CvBridge()

    def pre_perceive(self) -> bool:
        success = True
        # open gripper before picking
        success &= self.arm.execute_gripper_command(1.0)

        # go to perceive table pose
        # get pose from parameter server
        joint_angles_list = rospy.get_param("joint_angles")
        perceive_table_joint_angles = joint_angles_list["perceive_table"]

        success &= self.arm.send_joint_angles(perceive_table_joint_angles)

        return success

    def act(self) -> bool:
        success = True
        
        # subscribe to rgb 
        self.rgb_image = rospy.wait_for_message(self.rgb_topic, Image, timeout=10)

        self.process_image_with_cubes(self.rgb_image)

        return success

    def verify(self) -> bool:
        print("in verify")
        return True
    
    def process_image_with_cubes(self, image: Image):
        
        # convert image to cv2
        cv_image = self.bridge.imgmsg_to_cv2(image, desired_encoding='passthrough')

        # convert to bgr
        cv_image = cv2.cvtColor(cv_image, cv2.COLOR_RGB2BGR)

        # get the hsv_ranges from the parameter server
        hsv_ranges = rospy.get_param("clutter/hsv_ranges")

        # create colors list
        colors = [key for key in hsv_ranges.keys()]

        # Create HSV Image
        hsv_image = cv2.cvtColor(cv_image, cv2.COLOR_BGR2HSV)

        # create masks for each color
        masks = []
        for color in colors:
            lower = np.array(hsv_ranges[color]["lower"])
            upper = np.array(hsv_ranges[color]["upper"])
            mask = cv2.inRange(hsv_image, lower, upper)
            masks.append(mask)

        # get polygons for each color
        polygons = {}
        for i, color in enumerate(colors):
            polygons[color] = self.apply_mask_and_get_polygons(cv_image, masks[i])
        
        # combine all polygons
        polygons = [p for color in polygons for p in polygons[color]]
        
        draw_image = cv_image.copy()

        # check if two or more polygons are close to each other
        cluttered_polygons = self.check_polygons(polygons)

        if len(cluttered_polygons) > 0:

            # draw cluttered polygons
            for polygon in cluttered_polygons:
                # convert to contour
                polygon = np.array(polygon.exterior.coords, dtype=np.int32)
                # draw contour
                cv2.drawContours(draw_image, [polygon], -1, (0, 0, 255), 3)

            cv2.imshow("clutters", draw_image)
            cv2.waitKey(0)

    def check_polygons(self, polygons: List[Polygon]) -> List[MultiPolygon]:
        cluttered_polygons = []

        # min_clearance from parameter server
        min_clearance = rospy.get_param("clutter/min_clearance")

        # check if two or more polygons are close to each other
        for polygon in polygons:
            multipolygons = []
            for other_polygon in polygons:
                if polygon == other_polygon:
                    continue

                if polygon.distance(other_polygon) < min_clearance:
                    multipolygons.append(other_polygon)

            if len(multipolygons) > 0:
                # add polygon to multipolygon
                multipolygons.append(polygon)
                # convert to multipolygon
                multipolygons = MultiPolygon(multipolygons)
                # make the multipolygon into single polygon
                polygon = multipolygons.convex_hull
                # add multipolygon to list
                cluttered_polygons.append(polygon)

        # check if any of the multipolygons are same and remove duplicates
        if len(cluttered_polygons) > 0:
            cluttered_multipolygons = [cluttered_polygons[0]]
            for polygon in cluttered_polygons:
                if polygon not in cluttered_multipolygons:
                    cluttered_multipolygons.append(polygon)
        else:
            cluttered_multipolygons = []

        return cluttered_multipolygons

    def apply_mask_and_get_polygons(self, cv_image, mask) -> List[Polygon]:
        # apply mask
        res = cv2.bitwise_and(cv_image, cv_image, mask=mask)

        # find contours for detecting cubes
        contours, hierarchy = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        # remove small contours
        contours = [c for c in contours if cv2.contourArea(c) > 1000]

        # convert to polygons
        polygons = [cv2.approxPolyDP(c, 0.01 * cv2.arcLength(c, True), True) for c in contours]

        # smooth polygons
        polygons = [cv2.convexHull(p) for p in polygons]

        # convert to shapely polygons
        polygons = [Polygon(p.reshape((p.shape[0], p.shape[2]))) for p in polygons]

        return polygons