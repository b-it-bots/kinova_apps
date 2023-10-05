#!/usr/bin/env python3

from enum import Enum
import math
from typing import List, Set, Union
import rospy

from sensor_msgs.msg import Image, PointCloud2, JointState
from geometry_msgs.msg import PoseStamped, Quaternion, PoseArray

import kortex_driver.msg

import numpy as np
import cv2
import cv_bridge
import matplotlib.pyplot as plt

from copy import deepcopy

import tf

from sklearn.decomposition import PCA
from sklearn.cluster import DBSCAN

import sensor_msgs.point_cloud2 as pc2

import message_filters

from shapely.geometry import Polygon, MultiPolygon, Point, LineString
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
    "fork": "utensils",
    "knife": "utensils",
    "spoon": "utensils",
    "cutlery_container": "tableware",
    "square_plate": "tableware",
    "circle_plate": "tableware",
    "mini_fork": "tableware",
    "cup": "tableware",
    "bowl": "tableware",
    "broom": "cleaning",
    "dustpan": "cleaning",
    "brush": "cleaning",
    "toothbrush": "personal_care",
    "toothpaste": "personal_care",
    "shampoo": "personal_care",
    "soap": "personal_care",
}


class PropertyPolygon(Polygon):
    _id_to_attrs = {}

    __slots__ = Polygon.__slots__

    color: str
    mask: np.ndarray
    angle: float

    def __init__(self, exterior, holes=None, color=None, mask=None, angle=None):
        self._id_to_attrs[id(self)] = dict(color=color, mask=mask, angle=angle)

    def __new__(cls, exterior, holes=None, *args, **kwargs):
        poly = super().__new__(cls, exterior, holes)
        poly.__class__ = cls
        return poly

    def __del__(self):
        del self._id_to_attrs[id(self)]

    def __getattr__(self, key):
        try:
            return PropertyPolygon._id_to_attrs[id(self)][key]
        except KeyError as e:
            raise AttributeError(key) from e

    def __setattr__(self, key, value):
        if key in PropertyPolygon.__slots__:
            super().__setattr__(key, value)
        else:
            PropertyPolygon._id_to_attrs[id(self)][key] = value

    def __str__(self):
        return f"{self.color}"


class ClearClutterAction(AbstractAction):
    def __init__(
        self,
        arm: FullArmMovement,
        transform_utils: TransformUtils,
        object_type: ObjectType,
        reference_frame: str = "base_link",
        debug: bool = True,
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
        handle_model_name = rospy.get_param(node_name + "/handle_model_name")

        model = model_path + model_name
        handle_model = model_path + handle_model_name

        self.yolo_detector = YoloDetector(model, handle_model)

        self.pick_objects = rospy.get_param(node_name + "/pick_objects")

    def pre_perceive(self) -> bool:
        success = True
        # open gripper before picking
        success &= self.arm.execute_gripper_command(0.0)

        # sort place angles
        sort_place_angles = rospy.get_param("sort_place_poses")
        self.sort_place_joint_angles = sort_place_angles["data"]

        # success &= self.arm.send_joint_angles(perceive_table_joint_angles)

        return success
    
    def perceive_table(self):
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
            (
                has_clutter,
                detections,
                polygons,
            ) = self.process_image_with_cubes(self.rgb_image)


        elif self.object_type == ObjectType.REAL_OBJECTS:
            # for real objects
            # debug_image, polygons = self.yolo_detector.detect(self.rgb_image)

            (
                debug_image,
                classes,
                class_masks,
                polygons,
            ) = self.yolo_detector.detect_segments(self.rgb_image)


            # publish debug image
            self.debug_image_pub.publish(
                self.bridge.cv2_to_imgmsg(debug_image, encoding="passthrough")
            )

            detections = [(classes[i], class_masks[i]) for i in range(len(classes))]
            print(f"number of detections: {len(detections)}")

            if len(classes) == 0:
                return [], []

            big_objects = ['bowl', 'cup', 'square_plate', 'circle_plate', 'dustpan', 'brush', 'cutlery_container', 'soap']

            # sort the detections based on big objects list. detections can have other objects
            detections = sorted(detections, key=lambda x: big_objects.index(x[0].lower()) if x[0].lower() in big_objects else len(big_objects))

            for i in range(len(detections)):
                print(f"{detections[i][0]}")

        return detections, polygons

    def act(self) -> bool:
        success = True

        # get poses to perceive
        if self.object_type == ObjectType.CUBES:
            # pose_names = ['perceive_table_slant_left', 'perceive_table_slant_right']
            pose_names = ['perceive_table']
        elif self.object_type == ObjectType.REAL_OBJECTS:
            pose_names = [f'pose_{i}' for i in range(1, 4)]

        # get poses from parameter server
        percieve_poses = []
        joint_angles_list = rospy.get_param("joint_angles")
        for pose_name in pose_names:
            percieve_poses.append(joint_angles_list["data"][pose_name])

        for i, ppose in enumerate(percieve_poses):
            print(f'going to pose {pose_names[i]}')
            success &= self.arm.send_joint_angles(ppose)
            self.rgb_image = None
            self.pc = None

            # perceive table
            detections, polygons = self.perceive_table()

            if len(detections) == 0:
                continue

            # process point cloud
            print("processing point cloud")
            self.process_point_cloud()

            # get point cloud clusters
            polygon_rois, poses = self.get_point_cloud_clusters(detections)

            positions = None
            # if len(polygons) > 0:
            #     positions = self.get_pose_of_clutter(polygons)

            print(f"number of rois: {len(polygon_rois)}")

            # publish pose array
            if self.debug:
                # publish all the rois
                for i, roi in enumerate(polygon_rois):
                    pc = pc2.create_cloud_xyz32(
                        self.pc.header, roi.reshape((-1, 3))
                    )
                    self.pc_pub.publish(pc)
                    rospy.sleep(1)

                pose_array = PoseArray()
                pose_array.header.frame_id = self.reference_frame
                if self.object_type == ObjectType.CUBES:
                    pose_array.poses = [pose.pose for pose in poses]
                else:
                    pose_array.poses = [pose.pose for name, pose in poses]
                self.pose_array_pub.publish(pose_array)

            if self.pick_objects:
                # pick up the cubes one by one
                for i, pose in enumerate(poses):
                    if self.object_type == ObjectType.REAL_OBJECTS:
                        print()
                        print(f'picking up object {pose[0]}')

                        kpose: KinovaPose = get_kinovapose_from_pose_stamped(pose[1])
                    else:
                        kpose: KinovaPose = get_kinovapose_from_pose_stamped(pose)

                    if self.object_type == ObjectType.CUBES:
                        # open gripper
                        success &= self.arm.execute_gripper_command(0.55)
                    else:
                        object_name = pose[0]
                        # open gripper
                        if object_name == "cup" or object_name == "bowl" or object_name == "circle_plate":
                            success &= self.arm.execute_gripper_command(0.6)
                        elif object_name == "square_plate":
                            success &= self.arm.execute_gripper_command(0.5)
                        elif object_name == "soap":
                            success &= self.arm.execute_gripper_command(0.35)
                        elif object_name == "shampoo":
                            success &= self.arm.execute_gripper_command(0.45)
                        elif object_name == "dustpan" or object_name == "brush":
                            success &= self.arm.execute_gripper_command(0.55)
                        elif object_name == "toothbrush" or object_name == "fork" or object_name == "knife" or object_name == "spoon" or object_name == "mini_fork" or object_name == "broom":
                            success &= self.arm.execute_gripper_command(0.7)
                        else:
                            success &= self.arm.execute_gripper_command(0.5)

                    # adjust the z
                    # kpose.theta_z_deg += 180
                    if self.object_type == ObjectType.REAL_OBJECTS:
                        if (pose[0] == "cup"
                            or pose[0] == "bowl"
                            or pose[0] == "circle_plate" 
                            or pose[0] == "square_plate"):
                            kpose.z += 0.15
                        else:
                            kpose.z += 0.1
                        kpose.theta_z_deg += 90
                    else:
                        kpose.z += 0.1
                        kpose.theta_z_deg += 90

                    success &= self.arm.send_cartesian_pose(kpose)

                    # go down
                    if self.object_type == ObjectType.REAL_OBJECTS:
                        if (pose[0] == "cup"
                            or pose[0] == "bowl"
                            or pose[0] == "circle_plate" 
                            or pose[0] == "square_plate"):
                            kpose.z -= 0.15
                        else:
                            kpose.z -= 0.1
                    else:
                        kpose.z -= 0.1

                    # get current pose
                    current_pose = self.arm.get_current_pose()

                    if ((current_pose.x > 0.45 and current_pose.y > 0.15)
                        or (current_pose.x > 0.45 and current_pose.y < -0.19)
                        or current_pose.x > 0.5):
                        force_thresh = [6, 6, 6]
                    else:
                        force_thresh = [4, 4, 4]

                    if (
                        self.object_type == ObjectType.REAL_OBJECTS):
                        if (pose[0] == "cup"):
                            kpose.z += 0.045
                            success &= self.arm.send_cartesian_pose(kpose)
                            kpose.z += 0.15
                        elif pose[0] == "circle_plate":
                            kpose.z += 0.04
                            success &= self.arm.send_cartesian_pose(kpose)
                            kpose.z += 0.15
                        elif pose[0] == "bowl":
                            kpose.z += 0.055
                            success &= self.arm.send_cartesian_pose(kpose)
                            kpose.z += 0.15
                        elif pose[0] == "square_plate":
                            kpose.z = 0.05
                            success &= self.arm.send_cartesian_pose(kpose)
                            kpose.z += 0.15
                        elif pose[0] == "dustpan" and kpose.z > 0.025:
                            success &= self.arm.move_down_with_caution(
                                0.1,
                                force_threshold=force_thresh,
                                tool_z_thresh=kpose.z + 0.005,
                                retract_dist=0.04,
                            )
                            kpose.z += 0.15
                        elif pose[0] == "brush" and kpose.z > 0.025:
                            success &= self.arm.move_down_with_caution(
                                0.1,
                                force_threshold=force_thresh,
                                tool_z_thresh=kpose.z + 0.005,
                                retract_dist=0.005,
                            )
                            kpose.z += 0.15
                        elif pose[0] == "shampoo":
                            success &= self.arm.move_down_with_caution(
                                0.1,
                                force_threshold=force_thresh,
                                tool_z_thresh=kpose.z + 0.005,
                                retract_dist=0.03,
                            )
                            kpose.z += 0.15
                        elif pose[0] == "mini_fork":
                            success &= self.arm.move_down_with_caution(
                                0.1,
                                force_threshold=force_thresh,
                                tool_z_thresh=kpose.z + 0.005,
                                retract_dist=0.0,
                            )
                            kpose.z += 0.15
                        elif pose[0] == "fork" or pose[0] == "knife" or pose[0] == "spoon" or pose[0] == "toothbrush":
                            success &= self.arm.move_down_with_caution(
                                0.1,
                                force_threshold=force_thresh,
                                tool_z_thresh=kpose.z + 0.005,
                                retract_dist=0.005,
                            )
                            kpose.z += 0.15
                        else:
                            success &= self.arm.move_down_with_caution(
                                0.1,
                                force_threshold=force_thresh,
                                tool_z_thresh=kpose.z + 0.005,
                                retract_dist=0.01,
                            )
                            kpose.z += 0.15
                    else:
                        success &= self.arm.move_down_with_caution(
                            0.1,
                            force_threshold=[4, 4, 4],
                            tool_z_thresh=kpose.z + 0.005,
                            retract_dist=0.01,
                        )
                        kpose.z += 0.15

                    # close gripper to pick
                    success &= self.arm.execute_gripper_command(1.0)

                    # go up
                    success &= self.arm.send_cartesian_pose(kpose)

                    # go to sort place
                    # get the name of the object
                    if self.object_type == ObjectType.CUBES:
                        object_class = detections[i].color
                    elif self.object_type == ObjectType.REAL_OBJECTS:
                        object_name = pose[0]
                        object_class = object_map[object_name.lower()]

                    print(f"object class: {object_class}")

                    # get the joint angles for the object
                    # joint_angles = self.sort_place_joint_angles[object_class]
                    joint_angles = self.sort_place_joint_angles[object_class]

                    # send joint angles
                    success &= self.arm.send_joint_angles(joint_angles)

                    # open gripper
                    success &= self.arm.execute_gripper_command(0.0)

                    # get current pose
                    current_pose = self.arm.get_current_pose()

                    print(f"current pose: {current_pose}")

                    # if current pose x is less than 0, then move to safe pose
                    if current_pose.x < 0.1 and current_pose.y > 0:
                        print(f'moving to safe pose left')
                        success &= self.arm.send_joint_angles(
                            joint_angles_list["data"]["safe_pose_left"]
                        )
                    elif current_pose.x < 0.1 and current_pose.y < 0:
                        print(f'moving to safe pose right')
                        success &= self.arm.send_joint_angles(
                            joint_angles_list["data"]["safe_pose_right"]
                        )

                    print(f'object placed!')

                if len(polygons) > 0 and positions is not None and len(positions) > 0:
                    joint_angles_list = rospy.get_param("joint_angles")
                    pose = joint_angles_list["data"]["perceive_table"]
                    success &= self.arm.send_joint_angles(pose)

                    # get current pose
                    current_pose = self.arm.get_current_pose()

                    for position in positions:
                        ncp = deepcopy(current_pose)
                        ncp.x = position[0]
                        ncp.y = position[1]
                        ncp.z = position[2] + 0.02

                        success &= self.arm.execute_gripper_command(0.0)

                        success &= self.arm.send_cartesian_pose(ncp)

                        success &= self.arm.execute_gripper_command(0.5)

                        rospy.sleep(0.5)

                        ncp.z += 0.05

                        success &= self.arm.send_cartesian_pose(ncp)

                        success &= self.arm.execute_gripper_command(0.0)

        return success

    def verify(self) -> bool:

        print(f'verifying...')
        
        success = True

        success &= self.arm.execute_gripper_command(0.0)

        # get poses to perceive
        if self.object_type == ObjectType.CUBES:
            # pose_names = ['perceive_table_slant_left', 'perceive_table_slant_right']
            pose_names = ['perceive_table']
        elif self.object_type == ObjectType.REAL_OBJECTS:
            pose_names = [f'pose_{i}' for i in range(1, 4)]

        # get poses from parameter server
        percieve_poses = []
        joint_angles_list = rospy.get_param("joint_angles")
        for pose_name in pose_names:
            percieve_poses.append(joint_angles_list["data"][pose_name])

        poses_done = []
        for i, ppose in enumerate(percieve_poses):
            print(f'going to pose {pose_names[i]}')
            success &= self.arm.send_joint_angles(ppose)

            self.rgb_image = None
            self.pc = None
            detections, polygons = self.perceive_table()

            print(f'having {len(detections)} detections and {len(polygons)} polygons')

            if self.object_type == ObjectType.CUBES:
                if len(detections) == 0 and len(polygons) == 0:
                    poses_done.append(True)
                else:
                    poses_done.append(False)
            elif self.object_type == ObjectType.REAL_OBJECTS:
                if len(detections) == 0:
                    poses_done.append(True)
                else:
                    poses_done.append(False)

        print(f'poses done: {poses_done}')
        
        # if all are true then return true
        if all(poses_done):
            return True
        else:
            return False
        

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
        # cv2.imshow("pose", img)
        # cv2.waitKey(0)

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

    def get_pose_of_clutter(self, clutter_polys):

        positions = []

        cv_image = self.bridge.imgmsg_to_cv2(
            self.rgb_image, desired_encoding="passthrough"
        )

        for polys in clutter_polys:

            # convex hull
            polys_c = deepcopy(polys)
            poly = MultiPolygon(polys_c).convex_hull
            # convert to contour
            polyg_coords = np.array(poly.exterior.coords, dtype=np.int32)
            
            # create a mask from the polygons
            mask = np.zeros(cv_image.shape[:2], dtype=np.uint8)

            # draw contour
            cv2.drawContours(mask, [polyg_coords], -1, (255, 255, 255), -1)

            # filter the point cloud
            pc = self.pc_array[mask.astype(bool)]

            # flatten the point cloud
            fpc = pc.reshape((-1, 3))

            # remove nan values
            fpc = fpc[~np.isnan(fpc).any(axis=1)]

            # get the max z value and min z value of the point cloud
            max_z = np.max(fpc[:, 2])
            min_z = np.min(fpc[:, 2])

            # get the average z value of the point cloud
            middle_z = (max_z + min_z) / 2

            # get the mean of the point cloud
            mean = np.mean(fpc, axis=0)

            positions.append([mean[0], mean[1], middle_z])

        return positions
            

    def get_point_cloud_clusters(self, data):
        polys_pcs = []
        poses = []

        for element in data:
            if self.object_type == ObjectType.CUBES:
                # get the point cloud of the mask
                pc = self.pc_array[element.mask.astype(bool)]
            else:
                # get the point cloud from the mask
                pc = self.pc_array[element[1].astype(bool)]

            # flatten the point cloud
            fpc = pc.reshape((-1, 3))

            # remove nan values
            fpc = fpc[~np.isnan(fpc).any(axis=1)]

            # get the max z value and min z value of the point cloud
            max_z = np.max(fpc[:, 2])
            min_z = np.min(fpc[:, 2])

            if self.object_type == ObjectType.REAL_OBJECTS:
                print(f"element: {element[0]}")
                # if the element is 'cup'
                element_name = element[0].lower()
                if element_name == "cup" and max_z < 0.05:
                    continue
                if (
                    element_name == "cup"
                    or element_name == "bowl"
                    or element_name == "circle_plate"
                ):
                    # crop the point cloud from the top to remove the handle
                    if element_name == 'bowl' and max_z > 0.06:
                        # filter between 0.7 and 0.8
                        fpc = fpc[
                            np.logical_and(fpc[:, 2] > 0.06, fpc[:, 2] < 0.07)
                        ]
                    else:
                        fpc = fpc[fpc[:, 2] > max_z - 0.005]
                    middle_z = np.mean(fpc[:, 2]) - 0.02
                else:
                    # get the average z value of the point cloud
                    middle_z = (max_z + min_z) / 2
            else:
                print(f"element color: {element.color}, {element.angle}")
                # get the average z value of the point cloud
                middle_z = (max_z + min_z) / 2

            # get the mean of the point cloud
            if self.object_type == ObjectType.REAL_OBJECTS:
                if element_name == "circle_plate" or element_name == "bowl" or element_name == "cup":
                    
                    num_points = 0
                    while num_points < 4:
                        # get a random point from the point cloud
                        random_index = np.random.randint(0, len(fpc))
                        # get a cluster of points around the random point
                        fpc = fpc[
                            np.linalg.norm(fpc - fpc[random_index], axis=1) < 0.025
                        ]
                        num_points = len(fpc)
                    mean = np.mean(fpc, axis=0)
                elif element_name == "square_plate":
                    mean = np.mean(fpc, axis=0)
                    # offset the x and y
                    mean[1] += 0.075
                else:
                    mean = np.mean(fpc, axis=0)
            else:
                mean = np.mean(fpc, axis=0)

            # if points lie below the table, discard the object
            # if fpc[:, 2].min() < 0.0:
            #     continue

            if self.object_type == ObjectType.CUBES:
                angle = element.angle
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
            poses.append((element[0].lower(), pose))

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

        GRIPPER_LEFT_POINTS = np.array([[0, 614],[180, 514],[216, 502],[224, 682],[180, 720],[36, 720],[0, 694],[0, 626]])
        GRIPPER_RIGHT_POINTS = np.array([[1128, 720],[1076, 674],[1072, 498],[1120, 502],[1280, 582],[1280, 702],[1280, 720],[1128, 720]])
        cv2.fillPoly(cv_image, pts=[GRIPPER_LEFT_POINTS, GRIPPER_RIGHT_POINTS], color=(0, 0, 0))

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
        big_polygons = {}
        detected_masks = {}
        for i, color in enumerate(colors):
            dmask, polys, big_polys = self.apply_mask_and_get_masks_polygons(
                cv_image, masks[i], color
            )
            detected_masks[color] = dmask
            polygons[color] = polys
            big_polygons[color] = big_polys

        # combine all polygons
        ps = []
        for color in polygons:
            for p, m in zip(polygons[color], detected_masks[color]):
                p.color = color
                p.mask = m
                ps.append(p)

        draw_image = cv_image.copy()

        # check if two or more polygons are close to each other
        cluttered_polygons, free_polygons = self.check_polygons(ps)

        if len(cluttered_polygons) > 0:
            (
                can_be_picked_polys,
                new_clutter_polys,
            ) = self.analyze_cluttered_polys(cluttered_polygons)

            cluttered_polygons = new_clutter_polys

            free_polygons.extend(can_be_picked_polys)

            # draw cluttered polygons
            for polys in cluttered_polygons:
                
                # convex hull
                polys_c = deepcopy(polys)
                poly = MultiPolygon(polys_c).convex_hull
                # convert to contour
                polyg_coords = np.array(poly.exterior.coords, dtype=np.int32)
                # draw contour
                cv2.drawContours(draw_image, [polyg_coords], -1, (0, 0, 255), 3)

        dms = []
        for i, poly in enumerate(free_polygons):
            dmask = poly.mask
            poly_color = poly.color

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
                poly_color,
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

            if poly.angle is None:
                # get the angle of the box
                angle = np.arctan2(
                    side_middle[1] - center[1], side_middle[0] - center[0]
                )
            else:
                angle = poly.angle

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

            poly.angle = angle

            dms.append(poly)

        self.debug_image_pub.publish(
            self.bridge.cv2_to_imgmsg(draw_image, encoding="passthrough")
        )

        return len(cluttered_polygons) > 0, dms, cluttered_polygons

    def check_polygons(self, polygons: List[PropertyPolygon]):
        cluttered_polygons: List[List[PropertyPolygon]] = []
        free_polygons: List[PropertyPolygon] = []

        # min_clearance from parameter server
        min_clearance = rospy.get_param("clutter/min_clearance")

        cluttered_poly_indices = []

        # check if two or more polygons are close to each other
        for i, first_poly in enumerate(polygons):
            first_poly_clutters = []
            for j, second_poly in enumerate(polygons):
                if i == j:
                    continue

                if first_poly.distance(second_poly) < min_clearance:
                    if j not in cluttered_poly_indices:
                        first_poly_clutters.append(second_poly)
                        cluttered_poly_indices.append(j)

            if len(first_poly_clutters) > 0:
                first_poly_clutters.append(first_poly)
                cluttered_poly_indices.append(i)
                cluttered_polygons.append(first_poly_clutters)

        # get the free polygons
        free_poly_indices = [
            i for i in range(len(polygons)) if i not in cluttered_poly_indices
        ]

        free_polygons = [polygons[i] for i in free_poly_indices]

        return cluttered_polygons, free_polygons

    def apply_mask_and_get_masks_polygons(self, cv_image, mask, color):
        """
        apply the color mask on the cv image to detect cubes
        return the masks and shapely polygons for each cube
        """

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
        big_polygons = []
        for c in contours:
            # create mask
            mask = np.zeros(cv_image.shape[:2], np.uint8)
            # draw contour
            cv2.drawContours(mask, [c], -1, 255, -1)
            # add mask to list
            masks.append(mask)

            dmask = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)

            # get the contours from the mask
            # convert dmask to 8uc1
            dmask = cv2.cvtColor(dmask, cv2.COLOR_BGR2GRAY)
            contours, hierarchy = cv2.findContours(
                dmask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE
            )

            # remove small contours
            contours = [c for c in contours if cv2.contourArea(c) > 1000]
            big_contours = [c for c in contours if cv2.contourArea(c) > 3500 and cv2.contourArea(c) < 8000]

            # get the contour with the largest area
            contour = max(contours, key=cv2.contourArea)
            
            # fit a square to the contour
            rect = cv2.minAreaRect(contour)
            box = cv2.boxPoints(rect)
            box = np.int0(box)

            # convert to shapely polygon
            poly = PropertyPolygon(box)
            polygons.append(poly)

            if big_contours:
                big_contour = max(big_contours, key=cv2.contourArea)
                # fit a square to the contour
                big_rect = cv2.minAreaRect(big_contour)
                big_box = cv2.boxPoints(big_rect)
                big_box = np.int0(big_box)
                big_poly = PropertyPolygon(big_box)
                big_polygons.append(big_poly)

        return masks, polygons, big_polygons

    def analyze_cluttered_polys(
        self, clutter_polys
    ) -> Set[Union[List[PropertyPolygon], List[List[PropertyPolygon]]]]:
        can_be_picked_polys: List[PropertyPolygon] = []
        new_clutter_polys: List[List[PropertyPolygon]] = []

        for multi_poly in clutter_polys:
            local_clutter_polys = []
            for rect in multi_poly:

                # get the lines of opposite sides of the polygon
                opp_lines_1: List[LineString] = []
                opp_lines_2: List[LineString] = []

                l1 = LineString(
                    [rect.exterior.coords[0], rect.exterior.coords[1]]
                )
                l2 = LineString(
                    [rect.exterior.coords[2], rect.exterior.coords[3]]
                )
                opp_lines_1.append(l1)
                opp_lines_1.append(l2)

                l1 = LineString(
                    [rect.exterior.coords[1], rect.exterior.coords[2]]
                )
                l2 = LineString(
                    [rect.exterior.coords[3], rect.exterior.coords[0]]
                )
                opp_lines_2.append(l1)
                opp_lines_2.append(l2)

                ol1_free = True
                # check if each line in opp_lines_1 has min dist from other polygons in multi_poly
                for line in opp_lines_1:
                    # check if perpendicular distance is less than min_clearance
                    line_center = line.centroid

                    # find a point far away from the line center in the direction perpendicular to the line
                    l0 = line.parallel_offset(10, "right")

                    line_center = line.centroid
                    point1 = l0.centroid - line_center

                    point2 = (
                        line.parallel_offset(30, "right").centroid - line_center
                    )
                    point3 = (
                        line.parallel_offset(50, "right").centroid - line_center
                    )

                    for other_poly in multi_poly:
                        if other_poly == rect:
                            continue

                        # check if the point is inside the polygon
                        if (
                            other_poly.contains(point1)
                            or other_poly.contains(point2)
                            or other_poly.contains(point3)
                        ):
                            ol1_free = False
                            break

                    if not ol1_free:
                        break

                if ol1_free:
                    # find angle parallel to the parallel lines
                    angle = np.arctan2(
                        line.coords[1][1] - line.coords[0][1],
                        line.coords[1][0] - line.coords[0][0],
                    )

                    rect.angle = angle

                    can_be_picked_polys.append(rect)
                    continue

                ol2_free = True
                # check if each line in opp_lines_2 has min dist from other polygons in multi_poly
                for line in opp_lines_2:
                    l0 = line.parallel_offset(10, "right")

                    line_center = line.centroid
                    point1 = l0.centroid - line_center
                    point2 = (
                        line.parallel_offset(30, "right").centroid - line_center
                    )
                    point3 = (
                        line.parallel_offset(50, "right").centroid - line_center
                    )
                    for other_poly in multi_poly:
                        if other_poly == rect:
                            continue

                        # check if the point is inside the polygon
                        if (
                            other_poly.contains(point1)
                            or other_poly.contains(point2)
                            or other_poly.contains(point3)
                        ):
                            ol2_free = False
                            break

                    if not ol2_free:
                        break

                if ol2_free:
                    # find angle parallel to the parallel lines
                    angle = np.arctan2(
                        line.coords[1][1] - line.coords[0][1],
                        line.coords[1][0] - line.coords[0][0],
                    )

                    rect.angle = angle

                    can_be_picked_polys.append(rect)
                    continue

                if not ol1_free and not ol2_free:
                    local_clutter_polys.append(rect)

            if len(local_clutter_polys) > 0:
                new_clutter_polys.append(local_clutter_polys)

        return can_be_picked_polys, new_clutter_polys
