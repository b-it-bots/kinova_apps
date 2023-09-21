#!/usr/bin/env python3

import math
from typing import List
import rospy

from sensor_msgs.msg import Image, PointCloud2
from geometry_msgs.msg import PoseStamped, Quaternion, PoseArray

import kortex_driver.msg

import numpy as np
import cv2
import cv_bridge
import matplotlib.pyplot as plt

import tf

from sklearn.decomposition import PCA

import sensor_msgs.point_cloud2 as pc2

import message_filters

from shapely.geometry import Polygon, MultiPolygon, Point
from shapely import affinity, plotting

from kinova_apps.abstract_action import AbstractAction
from kinova_apps.full_arm_movement import FullArmMovement
from utils.transform_utils import TransformUtils
from utils.kinova_pose import KinovaPose, get_kinovapose_from_pose_stamped


class ClearClutterAction(AbstractAction):
    def __init__(
        self,
        arm: FullArmMovement,
        transform_utils: TransformUtils,
        reference_frame: str = "base_link",
        debug: bool = False,
    ) -> None:
        super().__init__(arm, transform_utils)

        self.reference_frame = reference_frame

        self.cartesian_velocity_pub = rospy.Publisher(
            "/my_gen3/in/cartesian_velocity", kortex_driver.msg.TwistCommand, queue_size=1
        )

        # publisher for transform point cloud
        self.pc_pub = rospy.Publisher("/transformed_point_cloud", PointCloud2, queue_size=10)

        # publisher for pose array
        self.pose_array_pub = rospy.Publisher("/pose_array", PoseArray, queue_size=10)

        self.rgb_topic = 'input_image_topic'
        self.pc_topic = 'input_pointcloud_topic'

        self.bridge = cv_bridge.CvBridge()

        self.debug = debug

        self.rgb_image = None
        self.pc = None

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

        # create a syncronizer for rgb and depth
        self.image_sub = message_filters.Subscriber(self.rgb_topic, Image)
        self.pc_sub = message_filters.Subscriber(self.pc_topic, PointCloud2)
        sync = message_filters.ApproximateTimeSynchronizer([self.image_sub, self.pc_sub], 10, 0.2, allow_headerless=True)
        sync.registerCallback(self.perceive)
        
        # subscribe to rgb 
        # self.rgb_image = rospy.wait_for_message(self.rgb_topic, Image, timeout=10)

        print('here')

        while self.rgb_image is None:
            rospy.sleep(0.1)

        has_clutter, polygons, clutter_polys = self.process_image_with_cubes(self.rgb_image)

        if has_clutter:
            pass
        #     self.attack_clutter(polygons)
        #    # go to perceive table pose
        #     re-perceive
        #     check for clutters

        # process point cloud
        print("processing point cloud")
        self.process_point_cloud()

        # get rois of the polygons
        # polygon_rois = self.get_point_clouds_of_polygons(polygons)
        polygon_rois, poses = self.get_point_clouds_in_polygons(polygons)

        print(f'number of rois: {len(polygon_rois)}')

        # publish all the rois
        # for i, roi in enumerate(polygon_rois):
        #     print(f'roi shape: {roi.shape}')
        #     pc = pc2.create_cloud_xyz32(self.pc.header, roi.reshape((-1, 3)))
        #     print(f'publishing roi {i}')
        #     self.pc_pub.publish(pc)
        #     rospy.sleep(1)

        # publish pose array
        if self.debug:
            pose_array = PoseArray()
            pose_array.header.frame_id = self.reference_frame
            pose_array.poses = [pose.pose for pose in poses]
            self.pose_array_pub.publish(pose_array)
            rospy.sleep(1)

        # pick up the cubes one by one
        for pose in poses:
            kpose: KinovaPose = get_kinovapose_from_pose_stamped(pose)
            # open gripper
            success &= self.arm.execute_gripper_command(0.0)

            # adjust the z
            kpose.z += 0.125
            success &= self.arm.send_cartesian_pose(kpose)

            # go down
            kpose.z -= 0.1
            success &= self.arm.send_cartesian_pose(kpose)

            # close gripper
            success &= self.arm.execute_gripper_command(1.0)

            # go up
            kpose.z += 0.1
            success &= self.arm.send_cartesian_pose(kpose)

            # open gripper
            success &= self.arm.execute_gripper_command(0.0)


        return success

    def verify(self) -> bool:
        print("in verify")
        return True
    
    def draw_pose_on_image(self, pose: np.ndarray, eigenvector: np.ndarray) -> None:
        img = self.rgb_image
        # convert img to cv2
        img = self.bridge.imgmsg_to_cv2(img, desired_encoding='passthrough')

        # draw the eigenvector on the image
        img = cv2.arrowedLine(
            img,
            (int(pose[0]), int(pose[1])),
            (int(pose[0] + eigenvector[0] * 100), int(pose[1] + eigenvector[1] * 100)),
            (0, 0, 255),
            2,
        )
        # draw the pose on the image
        img = cv2.circle(img, (int(pose[0]), int(pose[1])), 5, (0, 0, 255), -1)
        # show the image
        cv2.imshow("pose", img)
        cv2.waitKey(0)   
    
    def get_pose_of_polygon(self, polygon: Polygon) -> np.ndarray:
        '''
        Computes the pose of the polygon using PCA.
        '''
        coords = np.array(polygon.exterior.coords.xy).T

        # compute pca
        mean = np.mean(coords, axis=0)
        centered_coords = coords - mean
        cov = np.cov(centered_coords, rowvar=False)
        eigenvalues, eigenvectors = np.linalg.eigh(cov)

        # sort eigenvalues and eigenvectors
        idx = np.argsort(eigenvalues)[::-1]
        eigenvalues = eigenvalues[idx]
        eigenvectors = eigenvectors[:, idx]

        # get the eigenvector corresponding to the smallest eigenvalue
        eigenvector = eigenvectors[:, -1]

        # get the eigenvector corresponding to the largest eigenvalue
        # eigenvector = eigenvectors[:, 0]

        # get the angle of the eigenvector
        angle = np.arctan2(eigenvector[1], eigenvector[0])

        # get the pose of the polygon
        pose = np.array([mean[0], mean[1], angle])

        # convert the angle to 0 to 360 degrees range
        if angle < 0:
            angle += 2 * np.pi

        # flip the quadrants of the angle to match the image
        angle = 2 * np.pi - angle

        # print pose, angle in degrees
        print(f'pose: {pose}, angle: {angle * 180 / np.pi}')

        return pose, eigenvector
    
    def get_point_clouds_in_polygons(self, polygons):
        
        polys_pcs = []
        poses = []

        min_table_z = np.inf

        for polygon in polygons:
            # get the bounding box of the polygon
            x_min, y_min, x_max, y_max = polygon.bounds

            # get the point cloud of the bounding box
            pc = self.pc_array[int(y_min):int(y_max), int(x_min):int(x_max)]

            # flatten the point cloud
            fpc = pc.reshape((-1, 3))

            # remove nan values
            fpc = fpc[~np.isnan(fpc).any(axis=1)]

            # get the max z value and min z value of the point cloud
            max_z = np.max(fpc[:, 2])
            min_z = np.min(fpc[:, 2])
            min_table_z = min(min_z, min_table_z)

            # TODO: modify XYZ value calculation

            if min_table_z < 0.0:
                # adjust the z values
                max_z += abs(min_table_z)
                min_z += abs(min_table_z)

            # get the average z value of the point cloud
            middle_z = (max_z + min_z) / 2

            print(f"max_z: {max_z}, min_z: {min_z}, middle_z: {middle_z}")

            pca = PCA(n_components=3)
            pca.fit(fpc)

            # get the eigenvector corresponding to the smallest eigenvalue
            eigenvector = pca.components_[-1]

            # get the mean of the point cloud
            mean = np.mean(fpc, axis=0)

            # get the angle of the eigenvector
            angle = np.arctan2(eigenvector[1], eigenvector[0])

            # define pose
            pose = PoseStamped()
            pose.header.frame_id = self.reference_frame
            pose.pose.position.x = mean[0]
            pose.pose.position.y = mean[1]
            pose.pose.position.z = middle_z

            pose.pose.orientation = Quaternion(*tf.transformations.quaternion_from_euler(math.pi, 0, angle))

            # append to the list
            polys_pcs.append(pc)
            poses.append(pose)

        return polys_pcs, poses
    
    def process_point_cloud(self) -> None:
        if self.pc is None:
            return
        
        # transform point cloud to base_link frame
        self.pc = self.transform_utils.transform_point_cloud('camera_color_frame', self.reference_frame, self.pc)

        # convert point cloud to pcl 
        pc_array = np.array(list(pc2.read_points(self.pc, skip_nans=False, field_names=("x", "y", "z"))), dtype=np.float32)
        # reshape to image shape
        self.pc_array: np.ndarray = pc_array.reshape((self.rgb_image.height, self.rgb_image.width, 3))

        print(f"point cloud shape: {self.pc_array.shape}")

        # # downsample the point cloud
        # pc = pcl.PointCloud(pc_array)

        # # print shape of pcl point cloud
        # print(f"pcl point cloud shape: {pc.size}")

        # # create a voxel grid filter
        # vox = pc.make_voxel_grid_filter()

        # # set the leaf size
        # leaf_size = 0.01
        # vox.set_leaf_size(leaf_size, leaf_size, leaf_size)

        # # apply filter
        # pc = vox.filter()

        # # print shape of pcl point cloud
        # print(f"pcl point cloud shape: {pc.size}")

        # # convert to numpy array
        # self.pc_array = pc.to_array()

        # print(f"point cloud shape: {self.pc_array.shape}")

    def perceive(self, img_msg, pc_msg) -> bool:
        # unsubscribe from rgb and depth
        self.image_sub.unregister()
        self.pc_sub.unregister()

        # process image
        self.rgb_image = img_msg
        self.pc = pc_msg

    
    def attack_clutter(self, polygons: List[Polygon]) -> None:
        pass
    
    def process_image_with_cubes(self, image: Image) -> [bool, List[Polygon], List[Polygon]]:
        
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

            return True, polygons, cluttered_polygons
        
        return False, polygons, []

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