#!/usr/bin/env python3

from enum import Enum
import math
from typing import List, Set, Union
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

from kinova_apps.clutter_pick.object_detction import YoloDetector

# enum for object type
class ObjectType(Enum):
    CUBES = 0
    REAL_OBJECTS = 1

# map objects to classes
object_map = {
    'fork': 'utensils',
    'knife': 'utensils',
    'spoon': 'utensils',
    'cup': 'tableware',
    'bowl': 'tableware',
    'mug': 'tableware',
    'smallbroom': 'cleaning',
    'smalldustpan': 'cleaning',
    'smallbrush': 'cleaning',
    'toothbrush': 'personal_care',
    'toothpaste': 'personal_care',
    'shampoo': 'personal_care',
    'soap': 'personal_care'
}

class ClearClutterAction(AbstractAction):
    def __init__(
        self,
        arm: FullArmMovement,
        transform_utils: TransformUtils,
        object_type: ObjectType,
        reference_frame: str = "base_link",
        debug: bool = True
    ) -> None:
        super().__init__(arm, transform_utils)

        self.object_type = ObjectType(object_type)

        rospy.loginfo("Clear Clutter Action")
        # log the object type
        rospy.loginfo(f"object type: {self.object_type.name}")

        self.reference_frame = reference_frame

        self.cartesian_velocity_pub = rospy.Publisher(
            "/my_gen3/in/cartesian_velocity",
            kortex_driver.msg.TwistCommand,
            queue_size=1,
        )

        # publisher for debug image
        self.debug_image_pub = rospy.Publisher(
            "/debug_image", Image, queue_size=10
        )

        # publisher for transform point cloud
        self.pc_pub = rospy.Publisher("/pc_rois", PointCloud2, queue_size=10)

        # publisher for pose array
        self.pose_array_pub = rospy.Publisher(
            "/pose_array", PoseArray, queue_size=10
        )

        self.rgb_topic = "input_image_topic"
        self.pc_topic = "input_pointcloud_topic"

        self.bridge = cv_bridge.CvBridge()

        self.debug = debug

        self.rgb_image = None
        self.pc = None

        # get node name
        node_name = rospy.get_name()
        model_path = rospy.get_param(node_name + "/model_path")
        model_name = rospy.get_param(node_name + "/model_name")

        model = model_path + model_name

        self.yolo_detector = YoloDetector(model)

        self.pick_objects = rospy.get_param(node_name + "/pick_objects")

    def pre_perceive(self) -> bool:
        success = True
        # open gripper before picking
        success &= self.arm.execute_gripper_command(0.0)

        # go to perceive table pose
        # get pose from parameter server
        joint_angles_list = rospy.get_param("joint_angles")
        perceive_table_joint_angles = joint_angles_list["data"]["perceive_table_slant"]

        # sort place angles
        sort_place_angles = rospy.get_param("sort_place_poses")
        self.sort_place_joint_angles = sort_place_angles["data"]

        success &= self.arm.send_joint_angles(perceive_table_joint_angles)

        return success

    def act(self) -> bool:
        success = True

        # create a syncronizer for rgb and depth
        self.image_sub = message_filters.Subscriber(self.rgb_topic, Image)
        self.pc_sub = message_filters.Subscriber(self.pc_topic, PointCloud2)
        sync = message_filters.ApproximateTimeSynchronizer(
            [self.image_sub, self.pc_sub], 10, 0.2, allow_headerless=True
        )
        sync.registerCallback(self.perceive)

        # wait for rgb image and point cloud
        while self.rgb_image is None:
            rospy.sleep(0.1)

        # for cubes
        if self.object_type == ObjectType.CUBES:
            has_clutter, detections, clutter_polys = self.process_image_with_cubes(self.rgb_image)

            # if has_clutter:
            #     pass
            #     self.attack_clutter(polygons)
            #    # go to perceive table pose
            #     re-perceive
            #     check for clutters
        
        elif self.object_type == ObjectType.REAL_OBJECTS:
            # for real objects
            # debug_image, polygons = self.yolo_detector.detect(self.rgb_image)

            debug_image, detections, polygons = self.yolo_detector.detect_segments(
                self.rgb_image
            )

            print(f"number of segment masks: {len(detections)}")

            # publish debug image
            self.debug_image_pub.publish(
                self.bridge.cv2_to_imgmsg(debug_image, encoding="passthrough")
            )

        # process point cloud
        print("processing point cloud")
        self.process_point_cloud()

        # get point cloud clusters
        polygon_rois, poses = self.get_point_cloud_clusters(detections)

        print(f"number of rois: {len(polygon_rois)}")

        # publish pose array
        if self.debug:
            # publish all the rois
            for i, roi in enumerate(polygon_rois):
                pc = pc2.create_cloud_xyz32(self.pc.header, roi.reshape((-1, 3)))
                self.pc_pub.publish(pc)
                rospy.sleep(1)

            pose_array = PoseArray()
            pose_array.header.frame_id = self.reference_frame
            pose_array.poses = [pose.pose for pose in poses]
            self.pose_array_pub.publish(pose_array)
            
        if self.pick_objects:

            # pick up the cubes one by one
            for i, pose in enumerate(poses):
                kpose: KinovaPose = get_kinovapose_from_pose_stamped(pose)

                if self.object_type == ObjectType.CUBES:
                    # open gripper
                    success &= self.arm.execute_gripper_command(0.25)
                else:
                    # open gripper
                    success &= self.arm.execute_gripper_command(0.0)

                # adjust the z
                kpose.z += 0.13
                # add 90 degrees to the orientation
                kpose.theta_z_deg += 180
                success &= self.arm.send_cartesian_pose(kpose)

                # go down
                kpose.z -= 0.1
                success &= self.arm.send_cartesian_pose(kpose)

                # close gripper to pick
                success &= self.arm.execute_gripper_command(1.0)

                # go up
                kpose.z += 0.1
                success &= self.arm.send_cartesian_pose(kpose)

                # go to sort place
                # get the name of the object
                if self.object_type == ObjectType.CUBES:
                    object_class = detections[i][0]
                elif self.object_type == ObjectType.REAL_OBJECTS:
                    object_name = list(detections.keys())[i]
                    object_class = object_map[object_name]

                print(f"object class: {object_class}")

                # get the joint angles for the object
                # joint_angles = self.sort_place_joint_angles[object_class]
                joint_angles = self.sort_place_joint_angles['default']

                # send joint angles
                success &= self.arm.send_joint_angles(joint_angles)

                # open gripper
                success &= self.arm.execute_gripper_command(0.0)

        return success

    def verify(self) -> bool:
        print("in verify")
        return True

    def draw_pose_on_image(
        self, pose: np.ndarray, eigenvector: np.ndarray
    ) -> None:
        img = self.rgb_image
        # convert img to cv2
        img = self.bridge.imgmsg_to_cv2(img, desired_encoding="passthrough")

        # draw the eigenvector on the image
        img = cv2.arrowedLine(
            img,
            (int(pose[0]), int(pose[1])),
            (
                int(pose[0] + eigenvector[0] * 100),
                int(pose[1] + eigenvector[1] * 100),
            ),
            (0, 0, 255),
            2,
        )
        # draw the pose on the image
        img = cv2.circle(img, (int(pose[0]), int(pose[1])), 5, (0, 0, 255), -1)
        # show the image
        cv2.imshow("pose", img)
        cv2.waitKey(0)

    def get_pose_of_polygon(self, polygon: Polygon) -> np.ndarray:
        """
        Computes the pose of the polygon using PCA.
        """
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
        print(f"pose: {pose}, angle: {angle * 180 / np.pi}")

        return pose, eigenvector

    def get_point_cloud_clusters(self, data):
        polys_pcs = []
        poses = []

        for element in data:

            if self.object_type == ObjectType.CUBES:
                # get the point cloud of the mask
                print(f'getting pc from mask.')
                pc = self.pc_array[element[1].astype(bool)]
            else:
                # get the point cloud from the mask
                pc = self.pc_array[data[element].astype(bool)]

            # flatten the point cloud
            fpc = pc.reshape((-1, 3))

            # remove nan values
            fpc = fpc[~np.isnan(fpc).any(axis=1)]

            # get the max z value and min z value of the point cloud
            max_z = np.max(fpc[:, 2])
            min_z = np.min(fpc[:, 2])

            if self.object_type == ObjectType.REAL_OBJECTS:
                print(f"element: {element}")
                # if the element is 'cup'
                if element == "cup" or element == "bowl" or element == "mug":
                    # crop the point cloud from the top to remove the handle
                    fpc = fpc[fpc[:, 2] > max_z - 0.01]
                    middle_z = max_z
                else:
                    # get the average z value of the point cloud
                    middle_z = (max_z + min_z) / 2
            else:
                print(f'element color: {element[0]}')
                # get the average z value of the point cloud
                middle_z = (max_z + min_z) / 2

            # get the mean of the point cloud
            mean = np.mean(fpc, axis=0)

            if self.object_type == ObjectType.CUBES:
                angle = element[2]
                # transform the angle from cv coordinate system to real world coordinate system
                angle = 2 * np.pi - angle
                # convert the angle to 0 to 360 degrees range
                if angle < 0:
                    angle += 2 * np.pi
                # subtract 90 degrees from the angle
                angle -= np.pi / 2
            else:
                pca = PCA(n_components=3)
                pca.fit(fpc)

                # get the eigenvector corresponding to the smallest eigenvalue
                eigenvector = pca.components_[0]

                # get the angle of the eigenvector
                angle = np.arctan2(eigenvector[1], eigenvector[0])

            # define pose
            pose = PoseStamped()
            pose.header.frame_id = self.reference_frame
            pose.pose.position.x = mean[0]
            pose.pose.position.y = mean[1]
            pose.pose.position.z = middle_z

            pose.pose.orientation = Quaternion(
                *tf.transformations.quaternion_from_euler(math.pi, 0, angle)
            )

            # append to the list
            polys_pcs.append(fpc)
            poses.append(pose)

        return polys_pcs, poses

    def process_point_cloud(self) -> None:
        if self.pc is None:
            return

        # transform point cloud to base_link frame
        self.pc = self.transform_utils.transform_point_cloud(
            "camera_color_frame", self.reference_frame, self.pc
        )

        # convert point cloud to numpy array
        pc_array = np.array(
            list(
                pc2.read_points(
                    self.pc, skip_nans=False, field_names=("x", "y", "z")
                )
            ),
            dtype=np.float32,
        )
        # reshape to image shape
        self.pc_array: np.ndarray = pc_array.reshape(
            (self.rgb_image.height, self.rgb_image.width, 3)
        )

        print(f"point cloud shape: {self.pc_array.shape}")

    def perceive(self, img_msg, pc_msg) -> bool:
        # unsubscribe from rgb and depth
        self.image_sub.unregister()
        self.pc_sub.unregister()

        # process image
        self.rgb_image = img_msg
        self.pc = pc_msg

    def attack_clutter(self, polygons: List[Polygon]) -> None:
        pass

    def process_image_with_cubes(
        self, image: Image
    ) -> [bool, List[Polygon], List[Polygon]]:
        
        # convert image to cv2
        cv_image = self.bridge.imgmsg_to_cv2(
            image, desired_encoding="passthrough"
        )

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
        detected_masks = {}
        for i, color in enumerate(colors):
            dmask, polys = self.apply_mask_and_get_masks(cv_image, masks[i])
            detected_masks[color] = dmask
            polygons[color] = polys

        # combine all polygons
        polygons = [(color, p) for color in polygons for p in polygons[color]]
        detected_masks = [(color, dmask) for color in detected_masks for dmask in detected_masks[color]]

        draw_image = cv_image.copy()

        # check if two or more polygons are close to each other
        cluttered_polygons, free_poly_indices = self.check_polygons(polygons)


        if len(cluttered_polygons) > 0:
            # draw cluttered polygons
            for poly in cluttered_polygons:
                # convert to contour
                polyg_coords = np.array(poly.exterior.coords, dtype=np.int32)
                # draw contour
                cv2.drawContours(draw_image, [polyg_coords], -1, (0, 0, 255), 3)

            self.debug_image_pub.publish(
                self.bridge.cv2_to_imgmsg(draw_image, encoding="passthrough")
            )

        angles = []
        for i, (c, dmask) in enumerate(detected_masks):
            if i not in free_poly_indices:
                continue

            dmask = cv2.cvtColor(dmask, cv2.COLOR_GRAY2BGR)

            # get the contours from the mask
            # convert dmask to 8uc1
            dmask = cv2.cvtColor(dmask, cv2.COLOR_BGR2GRAY)
            contours, hierarchy = cv2.findContours(
                dmask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE
            )

            # remove small contours
            contours = [c for c in contours if cv2.contourArea(c) > 1000]

            # get the contour with the largest area
            contour = max(contours, key=cv2.contourArea)

            # fit a square to the contour
            rect = cv2.minAreaRect(contour)
            box = cv2.boxPoints(rect)
            box = np.int0(box)

            # draw the box
            cv2.drawContours(draw_image, [box], 0, (0, 0, 0), 2)

            # annotate the color of the box
            cv2.putText(
                draw_image,
                c,
                (int(rect[0][0]), int(rect[0][1])),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 0, 0),
                2,
            )

            # get the center of the box
            center = rect[0]

            # get two points corresponding to same side of the box
            p1 = box[1]
            p2 = box[2]

            side_middle = (p1 + p2) / 2

            # get the angle of the box
            angle = np.arctan2(side_middle[1] - center[1], side_middle[0] - center[0])
            
            # draw the pose on the image from box center
            draw_image = cv2.arrowedLine(
                draw_image,
                (int(center[0]), int(center[1])),
                (
                    int(center[0] + np.cos(angle) * 100),
                    int(center[1] + np.sin(angle) * 100),
                ),
                (0, 0, 255),
                2,
            )

            angles.append(angle)

        self.debug_image_pub.publish(
            self.bridge.cv2_to_imgmsg(draw_image, encoding="passthrough")
        )

        # add the angle to the detections
        detected_masks = [(cdmask[0], cdmask[1], angle) for cdmask, angle in zip(detected_masks, angles)]

        return len(cluttered_polygons)>0, detected_masks, cluttered_polygons

    def check_polygons(self, polygons: List[Set[Union[str, Polygon]]]) -> List[MultiPolygon]:
        cluttered_polygons = []

        # min_clearance from parameter server
        min_clearance = rospy.get_param("clutter/min_clearance")

        cluttered_poly_indices = []

        # check if two or more polygons are close to each other
        for i, (c, polygon) in enumerate(polygons):
            multipolygons = []
            for j, (oc, other_polygon) in enumerate(polygons):
                if i == j:
                    continue

                if polygon.distance(other_polygon) < min_clearance:
                    multipolygons.append(other_polygon)
                    cluttered_poly_indices.append(j)

            if len(multipolygons) > 0:
                # add polygon to multipolygon
                multipolygons.append(polygon)
                cluttered_poly_indices.append(i)
                # convert to multipolygon
                multipolygons = MultiPolygon(multipolygons)
                # make the multipolygon into single polygon
                poly = multipolygons.convex_hull
                # add multipolygon to list
                cluttered_polygons.append(poly)

        # get indices of non cluttered polygons
        non_cluttered_poly_indices = [i for i in range(len(polygons)) if i not in cluttered_poly_indices]

        # check if any of the multipolygons are same and remove duplicates
        if len(cluttered_polygons) > 0:
            cluttered_multipolygons = [cluttered_polygons[0]]
            for polyg in cluttered_polygons:
                if polyg not in cluttered_multipolygons:
                    cluttered_multipolygons.append(polyg)
        else:
            cluttered_multipolygons = []

        return cluttered_multipolygons, non_cluttered_poly_indices

    def apply_mask_and_get_polygons(self, cv_image, mask) -> List[Polygon]:
        # apply mask
        res = cv2.bitwise_and(cv_image, cv_image, mask=mask)

        # find contours for detecting cubes
        contours, hierarchy = cv2.findContours(
            mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE
        )

        # remove small contours
        contours = [c for c in contours if cv2.contourArea(c) > 1000]

        # convert to polygons
        polygons = [
            cv2.approxPolyDP(c, 0.01 * cv2.arcLength(c, True), True)
            for c in contours
        ]

        # smooth polygons
        polygons = [cv2.convexHull(p) for p in polygons]

        # convert to shapely polygons
        polygons = [
            Polygon(p.reshape((p.shape[0], p.shape[2]))) for p in polygons
        ]

        return polygons

    def apply_mask_and_get_masks(self, cv_image, mask):
        '''
        apply the color mask on the cv image to detect cubes
        return the masks for each cube
        '''
        
        # apply mask
        res = cv2.bitwise_and(cv_image, cv_image, mask=mask)

        # find contours for detecting cubes
        contours, hierarchy = cv2.findContours(
            mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE
        )

        # remove small contours
        contours = [c for c in contours if cv2.contourArea(c) > 1000]

        # get the pixels lying inside the contours
        masks = []
        polygons = []
        for c in contours:
            # create mask
            mask = np.zeros(cv_image.shape[:2], np.uint8)
            # draw contour
            cv2.drawContours(mask, [c], -1, 255, -1)
            # add mask to list
            masks.append(mask)
            # convert to shapely polygon
            polygons.append(Polygon(c.reshape((c.shape[0], c.shape[2]))))

        return masks, polygons