#!/usr/bin/env python3

import rospy
from kinova_apps.abstract_action import AbstractAction
from geometry_msgs.msg import PoseStamped, Quaternion
import geometry_msgs.msg
import sensor_msgs.msg
from sensor_msgs.msg import Image
from kinova_apps.full_arm_movement import FullArmMovement
from utils.transform_utils import TransformUtils
from utils.kinova_pose import (
    KinovaPose,
    get_kinovapose_from_list,
    get_kinovapose_from_pose_stamped,
)
from utils.perception_utils import (
    dilate,
    erode,
    get_uppermost_contour,
    detect_door_circle,
)

from kortex_driver.srv import *
from kortex_driver.msg import *

import tf
import numpy as np
import sensor_msgs.msg
from cv_bridge import CvBridge, CvBridgeError
import cv2
import pdb
from scipy.spatial.distance import cdist
import math
import os
import datetime
import yolov5


class ProbeAction(AbstractAction):
    def __init__(
        self, arm: FullArmMovement, transform_utils: TransformUtils
    ) -> None:
        super().__init__(arm, transform_utils)
        self.current_force_z = []
        self.current_height = None
        self.bridge = CvBridge()
        self.loop_rate = rospy.Rate(10)
        self.cart_vel_pub = rospy.Publisher(
            "/my_gen3/in/cartesian_velocity",
            kortex_driver.msg.TwistCommand,
            queue_size=1,
        )
        self.base_feedback_sub = rospy.Subscriber(
            "/my_gen3/base_feedback",
            kortex_driver.msg.BaseCyclic_Feedback,
            self.base_feedback_cb,
        )
        self.door_knob_pose_pub = rospy.Publisher(
            "/door_knob_pose", PoseStamped, queue_size=1
        )
        self.probe_cable_dir_debug_pub = rospy.Publisher(
            "/probe_cable_dir_debug", Image, queue_size=1
        )
        self.visual_servo_debug_img = rospy.Publisher(
            "/visual_servoing_debug_img", Image, queue_size=1
        )
        self.image_sub = rospy.Subscriber(
            "/camera/color/image_raw", sensor_msgs.msg.Image, self.image_cb
        )
        self.model = yolov5.load(
            "/home/b-it-bots/robothon_ros_workspace/src/kinova_apps/models/door_knob/door_knob_nano_ver1.pt"
        )
        self.model_params()
        self.save_debug_image_dir = "/home/b-it-bots/temp/robothon/door_knob"

        self.debug = rospy.get_param("~debug", False)
        self.transform_utils = TransformUtils()

    def base_feedback_cb(self, msg):
        self.current_force_z.append(msg.base.tool_external_wrench_force_z)
        if len(self.current_force_z) > 25:
            self.current_force_z.pop(0)
        self.current_height = msg.base.tool_pose_z

    def image_cb(self, msg):
        self.image = self.bridge.imgmsg_to_cv2(
            msg, desired_encoding="passthrough"
        )

    def pre_perceive(self) -> bool:
        rospy.loginfo("in pre perceive")

        return True

    def act(self) -> bool:
        """
        0: open door
        1: pluck and place in holder
        2: pick probe from holder
        3: probe circuit
        4: place probe safe
        """
        task_seq = [0, 1, 2, 3, 4]
        success = False
        if 0 in task_seq:
            # TODO: navigate safely to the next pose (it might hit the cupboard)
            # add a waypoint

            # open the door
            success = self.open_door_with_trajactroy()
            if not success:
                rospy.logerr("[probe_action] Failed to open the door")
                return False
        if 1 in task_seq:
            # pick the probe from the box and place it in the holder
            # TODO: update the stuff for the below method
            success = self.pluck_place_probe_in_holder()
            success = True
            if not success:
                rospy.logerr(
                    "[probe_action] Failed to place the probe in the holder"
                )
                return False

        if 2 in task_seq:
            # # pick the probe from the holder
            # # TODO: update the stuff for the below method
            success = self.pick_probe_from_holder()
        if 3 in task_seq:
            # probe the circuit
            success = self.probe_circuit()
            if not success:
                rospy.logerr("[probe_action] Failed to probe the circuit")
                return False

        if 4 in task_seq:
            # place the probe somewhere safe
            success = self.place_probe_safe()

            if not success:
                rospy.logerr(
                    "[probe_action] Failed to place the probe somewhere safe"
                )
                return False

        return success

    def verify(self) -> bool:
        rospy.loginfo("in verify")
        return True

    def pluck_place_probe_in_holder(self):
        """
        Pluck the cable probe from the box and place it in the holder
        """

        # go the probe initial position
        success = self.arm.execute_gripper_command(0.6)  # open the gripper

        # pluck the probe from the box
        success = self.pluck_probe_from_box()

        if not success:
            rospy.logerr(
                "[probe_action] Failed to pluck the probe from the box"
            )
            return False

        # place the probe in the holder
        success = self.place_probe_in_holder()

        if not success:
            rospy.logerr(
                "[probe_action] Failed to place the probe in the holder"
            )
            return False

        return True

    def pluck_probe_from_box(self):
        # get the probe initial position from tf
        msg = PoseStamped()
        msg.header.frame_id = "probe_initial_link"
        msg.header.stamp = rospy.Time(0)
        # TODO: check if this offset is correct wrt placing in holder
        probe_initial_pose = self.transform_utils.transformed_pose_with_retries(
            msg, "base_link", execute_arm=True, offset=[0, 0, -math.pi / 2]
        )

        # convert the probe initial position to a kinova pose
        probe_initial_pose_kp = get_kinovapose_from_pose_stamped(
            probe_initial_pose
        )

        # increase the z position by 5cm
        probe_initial_pose_kp.z += 0.08

        # send the probe initial position to the arm
        rospy.loginfo("[probe_action] moving to probe initial position")
        success = self.arm.send_cartesian_pose(probe_initial_pose_kp)

        if not success:
            rospy.logerr(
                "[probe_action] Failed to move to the probe initial position"
            )
            return False

        rospy.loginfo("[probe_action] reached probe initial position")

        self.arm.execute_gripper_command(0.6)  # open the gripper

        # use velocity control to move the probe down
        rospy.loginfo("[probe_action] moving down the probe")
        # success = self.arm.move_down_with_caution(force_threshold=[4,4,1.75], velocity=0.005, tool_z_thresh=0.10, retract_dist=0.008)

        probe_initial_pose_kp.z = 0.1173

        success = self.arm.send_cartesian_pose(probe_initial_pose_kp)

        if not success:
            rospy.logerr("[probe_action] Failed to move down the probe")
            return False

        rospy.loginfo("[probe_action] moved down the probe")

        # close the gripper
        success = self.arm.execute_gripper_command(1.0)

        if not success:
            rospy.logerr("Failed to close the gripper")
            return False

        # move the probe back in x direction for 4cm
        success = self.arm.move_with_velocity(0.025, 3, "y")

        if not success:
            rospy.logerr("Failed to move back the probe")
            return False

        probe_current_pose = self.arm.get_current_pose()

        probe_current_pose.z += 0.15

        rospy.loginfo("[probe_action] moving up the probe")
        success = self.arm.send_cartesian_pose(probe_current_pose)

        if not success:
            rospy.logerr("Failed to move up the probe")
            return False

        rospy.loginfo("[probe_action] moved up the probe")

        return True

    def place_probe_in_holder(self):
        # move the probe to the holder
        # TODO: test whether joint andles or cartesian pose is better

        probe_place_pre_holder_pose = rospy.get_param(
            "~probe_action_poses/probe_place_pre_holder_pose"
        )

        probe_place_pre_holder_pose_kp = get_kinovapose_from_list(
            probe_place_pre_holder_pose
        )

        rospy.loginfo(
            "[probe_action] moving to probe place pre holder position"
        )
        success = self.arm.send_cartesian_pose(
            probe_place_pre_holder_pose_kp, max_lin_vel=0.05
        )

        if not success:
            rospy.logerr(
                "[probe_action] Failed to move to the probe place pre holder position"
            )
            return False

        rospy.loginfo("[probe_action] reached probe place pre holder position")

        # get the probe holder position from rosparam
        # probe_place_pre_holder_pose_joint_angles = rospy.get_param("~joint_angles/probe_place_pre_holder_pose")

        # rospy.loginfo("[probe_action] sending joint angles")
        # success = self.arm.send_joint_angles(probe_place_pre_holder_pose_joint_angles)

        # if not success:
        #     rospy.logerr("Failed to move up the probe")
        #     return False

        # rospy.loginfo("moved the probe to above the holder")

        # place in the holder
        # probe_place_in_holder_pose_joint_angles = rospy.get_param("~joint_angles/probe_place_in_holder_pose")

        # rospy.loginfo("[probe_action] sending joint angles")
        # success = self.arm.send_joint_angles(probe_place_in_holder_pose_joint_angles)

        # if not success:
        #     rospy.logerr("Failed to move up the probe")
        #     return False

        # rospy.loginfo("moved the probe to above the holder")

        # do with cartesian control
        probe_place_in_holder_pose = rospy.get_param(
            "~probe_action_poses/probe_place_in_holder_pose"
        )

        probe_place_in_holder_pose_kp = get_kinovapose_from_list(
            probe_place_in_holder_pose
        )

        rospy.loginfo("[probe_action] moving to probe place holder position")
        success = self.arm.send_cartesian_pose(
            probe_place_in_holder_pose_kp, max_lin_vel=0.05
        )

        if not success:
            rospy.logerr(
                "[probe_action] Failed to move to the probe place holder position"
            )
            return False

        # open the gripper
        success = self.arm.execute_gripper_command(0.35)

        if not success:
            rospy.logerr("Failed to open the gripper")
            return False

        rospy.loginfo(
            "[probe_action] moving to probe place pre holder position"
        )
        success = self.arm.send_cartesian_pose(
            probe_place_pre_holder_pose_kp, max_lin_vel=0.05
        )

        if not success:
            rospy.logerr(
                "[probe_action] Failed to move to the probe place pre holder position"
            )
            return False

        rospy.loginfo("[probe_action] reached probe place pre holder position")

        safe_pose_after_probe_placement = rospy.get_param(
            "~probe_action_poses/safe_pose_after_probe_placement"
        )
        safe_pose_after_probe_placement_kp = get_kinovapose_from_list(
            safe_pose_after_probe_placement
        )
        rospy.loginfo("[probe_action] moving to pose after probe placement")
        success = self.arm.send_cartesian_pose(
            safe_pose_after_probe_placement_kp, max_lin_vel=0.05
        )

        if not success:
            rospy.logerr(
                "[probe_action] Failed to move to the probe place pre holder position"
            )
            return False

        rospy.loginfo("[probe_action] reached probe place pre holder position")

        return True

    def create_twist_from_velocity(self, velocity) -> Twist:
        """
        Create Twist message from velocity vector

        input: velocity vector :: np.array
        output: velocity vector :: Twist
        """
        board_wrt_base = self.transform_utils.get_pose_from_link(
            "base_link", "board_link"
        )

        # convert pose.orientation to yaw
        orientation = board_wrt_base.pose.orientation
        quaternion = (
            orientation.x,
            orientation.y,
            orientation.z,
            orientation.w,
        )
        euler = tf.transformations.euler_from_quaternion(quaternion)
        yaw = euler[2]

        velocity_vector = geometry_msgs.msg.Twist()
        velocity_vector.linear.x = velocity * math.cos(yaw)
        velocity_vector.linear.y = velocity * math.sin(yaw)
        velocity_vector.linear.z = 0

        velocity_vector.angular.x = 0
        velocity_vector.angular.y = 0
        velocity_vector.angular.z = 0

        return velocity_vector

    def open_door_with_trajactroy(self):  # working code
        enable_vs = False

        msg = PoseStamped()
        msg.header.frame_id = "door_knob_link"
        msg.header.stamp = rospy.Time(0)
        if enable_vs:
            msg.pose.position.x += 0.015
        else:
            msg.pose.position.x += 0.005

        door_knob_pose = self.transform_utils.transformed_pose_with_retries(
            msg, "base_link", execute_arm=True, offset=[0.0, 0.0, -math.pi / 2]
        )

        # convert the door knob pose to a kinova pose
        door_knob_kinova_pose = get_kinovapose_from_pose_stamped(door_knob_pose)

        print(
            "door knob pose =>> before",
            (door_knob_kinova_pose.x, door_knob_kinova_pose.y),
        )
        # move up a bit
        door_knob_kinova_pose.z += 0.022 + 0.05  # adding 5 cm for approach

        # send the door knob pose to the arm
        rospy.loginfo("[probe_action] moving to door knob position")
        rospy.loginfo(
            "[probe_action] door knob pose: {}".format(door_knob_kinova_pose)
        )

        success = self.arm.send_cartesian_pose(door_knob_kinova_pose)

        success = self.arm.execute_gripper_command(0.5)

        if not enable_vs:
            board_height = rospy.get_param("/board_height", 0.1157)
            door_knob_kinova_pose.z = board_height + 0.003
            success = self.arm.send_cartesian_pose(door_knob_kinova_pose)
            # return False

            if not success:
                rospy.logerr("Failed to move to the door knob position")
                return False

        if enable_vs:
            rospy.loginfo("Visual servoing to door knob")
            success = self.run_visual_servoing(
                self.get_door_knob_error_2,
                target_height=door_knob_kinova_pose.z,
                save_debug_images=True,
            )

            # record the current tool pose
            current_tool_pose = self.arm.get_current_pose()

            board_height = rospy.get_param("/board_height", 0.1157)
            current_tool_pose.z = board_height + 0.005
            success = self.arm.send_cartesian_pose(door_knob_kinova_pose)

            msg = PoseStamped()
            msg.header.frame_id = "door_knob_link"
            msg.header.stamp = rospy.Time(0)
            door_knob_pose = self.transform_utils.transformed_pose_with_retries(
                msg,
                "base_link",
                execute_arm=True,
                offset=[0.0, 0.0, -math.pi / 2],
            )
            door_knob_kinova_pose = get_kinovapose_from_pose_stamped(
                door_knob_pose
            )

            x_offset = (
                current_tool_pose.x - door_knob_kinova_pose.x
            )  # compensating for the 1.5cm addition before visual servoing
            y_offset = current_tool_pose.y - door_knob_kinova_pose.y

            print("---" * 10)
            print(
                "door knob pose ==>> after ",
                (door_knob_kinova_pose.x, door_knob_kinova_pose.y),
            )
            print(
                "current tool pose", (current_tool_pose.x, current_tool_pose.y)
            )
            print("x and y offset", (x_offset, y_offset))
            print("---" * 10)

            if success:
                rospy.loginfo("Visual servoing to door knob succeeded")
                pose_list = self.get_trajactory_poses(num_poses=8)
                for pose in pose_list:
                    pose.x += x_offset
                    pose.y += y_offset

        pose_list = self.get_trajactory_poses(num_poses=8)

        # Half closing the gripper before opening door
        # success = self.arm.execute_gripper_command(0.5)

        # rospy.sleep(1.0) # wait for the arm to settle for proper force sensing
        """
        current_tool_pose.z -= 0.02
        self.arm.send_cartesian_pose(current_tool_pose)

        # velocity mode to approach the door knob
        success = self.arm.move_down_with_caution(velocity=0.01, force_threshold=[4,4,2.75], retract_dist=0.0075)

        if not success:
            rospy.logerr("Failed to move down the arm")
            return False
        rospy.loginfo("[probe_action] reached door knob position")


        # get the current tool pose and create a tf link for it
        """

        # close the gripper
        success = self.arm.execute_gripper_command(0.90)  # 0.75 closed
        if not success:
            rospy.logerr("Failed to open the gripper")
            return False

        success = self.arm.traverse_waypoints(pose_list)

        if not success:
            rospy.logerr("Failed to reach desired pose")
            return False

        success = self.arm.execute_gripper_command(0.50)  # 0.75 closed
        if not success:
            rospy.logerr("Failed to open the gripper")
            return False

        # move up a bit
        current_pose = self.arm.get_current_pose()
        current_pose.z += 0.07

        success = self.arm.send_cartesian_pose(current_pose)
        if success:
            rospy.loginfo("Door knob opened successfully")

        return success

    def pick_probe_from_holder(self):
        # # go to the probe pick perceive position above the holder
        self.arm.execute_gripper_command(0.4)
        # probe_holder_pick_perceive_pose = rospy.get_param("~probe_action_poses/probe_holder_perceive_pose")
        # probe_holder_pick_perceive_pose_kinova_pose = get_kinovapose_from_list(probe_holder_pick_perceive_pose)

        # rospy.loginfo("[probe_action] moving to probe holder pick perceive position")
        # success = self.arm.send_cartesian_pose(probe_holder_pick_perceive_pose_kinova_pose)

        # if not success:
        #     rospy.logerr("[probe_action] Failed to move to the probe holder pick perceive position")
        #     return False

        # rospy.loginfo("[probe_action] reached probe holder pick perceive position")

        # get an image from the camera
        # image_msg = rospy.wait_for_message("/camera/color/image_raw", Image)
        # image = self.bridge.imgmsg_to_cv2(image_msg, "bgr8")

        # # get the probe cable direction
        # probe_cable_dir = self.get_probe_cable_dir(image)

        # rospy.loginfo(f'probe_cable_dir: {probe_cable_dir}')

        # if probe_cable_dir > 0:
        #     probe_cable_dir += 90
        # elif probe_cable_dir < 0:
        #     probe_cable_dir = -probe_cable_dir

        # get the probe pick from holder pose
        probe_pre_pick_from_holder_pose = rospy.get_param(
            "~probe_action_poses/probe_holder_pre_pick_pose"
        )
        probe_pre_pick_from_holder_pose_kinova_pose = get_kinovapose_from_list(
            probe_pre_pick_from_holder_pose
        )

        # probe_pre_pick_from_holder_pose_kinova_pose.theta_z_deg += 180.0

        # move to the pre pick pose
        rospy.loginfo("[probe_action] moving to probe pre pick position")
        success = self.arm.send_cartesian_pose(
            probe_pre_pick_from_holder_pose_kinova_pose
        )

        if not success:
            rospy.logerr(
                "[probe_action] Failed to move to the probe pre pick position"
            )
            return False

        rospy.loginfo("[probe_action] reached probe pre pick position")

        # rotate the arm to the probe cable direction and set the z to the pick pose z
        probe_pick_pose_from_holder = rospy.get_param(
            "~probe_action_poses/probe_holder_pick_pose"
        )
        probe_pick_pose_from_holder = get_kinovapose_from_list(
            probe_pick_pose_from_holder
        )
        # probe_pick_pose_from_holder.theta_z_deg += 180.0

        # move to the probe pick from holder pose
        rospy.loginfo(
            "[probe_action] moving to pick probe from holder position"
        )

        success = self.arm.send_cartesian_pose(probe_pick_pose_from_holder)

        if not success:
            rospy.logerr(
                "Failed to move to the pick probe from holder position"
            )
            return False

        rospy.loginfo(
            "[probe_action] reached to pick probe from holder position"
        )

        # close the gripper
        self.arm.execute_gripper_command(1.0)

        if not success:
            rospy.logerr("[probe_action] Failed to close the gripper")
            return False

        probe_pick_pose_from_holder = rospy.get_param(
            "~probe_action_poses/probe_holder_pick_pose"
        )
        probe_pick_pose_from_holder = get_kinovapose_from_list(
            probe_pick_pose_from_holder
        )
        probe_pick_pose_from_holder.z += 0.1

        success = self.arm.send_cartesian_pose(probe_pick_pose_from_holder)

        if not success:
            rospy.logerr("[probe_action] Failed to move up the probe")
            return False

        rospy.loginfo("[probe_action] probe picked")

        return True

    def probe_circuit(self):
        kinova_pose = self.transform_utils.transform_pose_frame_name(
            reference_frame_name="probe_circuit_link",
            target_frame_name="base_link",
            offset_linear=[0.0, 0.0, 0.3],
            offset_rotation_euler=[math.pi, 0.0, math.pi / 2],
        )
        self.arm.send_cartesian_pose(kinova_pose)

        probed = False
        retries = 0
        max_probe_retries = rospy.get_param("~max_probe_retries", 5)
        while not probed:
            rospy.loginfo("Trying probe for %d th time" % retries)
            self.run_visual_servoing(
                self.get_probe_point_error,
                target_height=0.3,
                save_debug_images=False,
            )
            probed = self.move_down_and_probe()
            rospy.loginfo("Probed: %d" % probed)
            retries += 1
            if retries > max_probe_retries:
                break

        # moving to safe height to avoid hitting the door
        current_pose = self.arm.get_current_pose()
        current_pose.z += 0.06
        self.arm.send_cartesian_pose(current_pose)

        return probed

    def place_probe_safe(self):
        # get the probe pick from holder pose
        probe_pre_pick_from_holder_pose = rospy.get_param(
            "~probe_action_poses/probe_holder_pre_pick_pose"
        )
        probe_pre_pick_from_holder_pose_kinova_pose = get_kinovapose_from_list(
            probe_pre_pick_from_holder_pose
        )

        # move to the pre pick pose
        rospy.loginfo("[probe_action] moving to probe pre pick position")
        success = self.arm.send_cartesian_pose(
            probe_pre_pick_from_holder_pose_kinova_pose
        )

        if not success:
            rospy.logerr(
                "[probe_action] Failed to move to the probe pre pick position"
            )
            return False

        # rotate the arm to the probe cable direction and set the z to the pick pose z
        probe_pick_pose_from_holder = rospy.get_param(
            "~probe_action_poses/probe_holder_pick_pose"
        )
        probe_pick_pose_from_holder = get_kinovapose_from_list(
            probe_pick_pose_from_holder
        )
        probe_pick_pose_from_holder.theta_z_deg = 90.0
        probe_pick_pose_from_holder.z += (
            0.05  # Dont go down to pick pose for placement
        )

        # move to the probe pick from holder pose
        rospy.loginfo(
            "[probe_action] moving to pick probe from holder position"
        )

        success = self.arm.send_cartesian_pose(probe_pick_pose_from_holder)

        if not success:
            rospy.logerr(
                "Failed to move to the pick probe from holder position"
            )
            return False

        rospy.loginfo(
            "[probe_action] reached to pick probe from holder position"
        )

        self.arm.execute_gripper_command(0.6)

        safe_pose_after_probe_placement = rospy.get_param(
            "~probe_action_poses/safe_pose_after_probe_placement"
        )
        safe_pose_after_probe_placement_kp = get_kinovapose_from_list(
            safe_pose_after_probe_placement
        )
        rospy.loginfo("[probe_action] moving to safe pose")
        success = self.arm.send_cartesian_pose(
            safe_pose_after_probe_placement_kp, max_lin_vel=0.1
        )

        if not success:
            rospy.logerr(
                "[probe_action] Failed to move to the probe place pre holder position"
            )
            return False

        currentPose = self.arm.get_current_pose()
        currentPose.z += 0.05
        success = self.arm.send_cartesian_pose(currentPose, max_lin_vel=0.1)

        if not success:
            rospy.logerr(
                "[probe_action] Failed to move to the probe place pre holder above position"
            )
            return False

        rospy.loginfo(
            "[probe_action] reached probe place pre holder above position"
        )

        return True

    def get_probe_cable_dir(self, image):
        """
        This function takes an image of the probe cable and returns the direction of the cable
        return: angle in degrees (in cv coordinate system)
        """

        rospy.logwarn("entered get probe")

        # crop the border of the image by 25%
        image = image[
            int(image.shape[0] * 0.25) : int(image.shape[0] * 0.75),
            int(image.shape[1] * 0.25) : int(image.shape[1] * 0.75),
        ]
        or_image = image.copy()

        # Convert the image to HSV color space
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

        # Define the lower and upper threshold for the pale orange color
        lower_orange = np.array([0, 100, 100], dtype=np.uint8)
        upper_orange = np.array([30, 255, 255], dtype=np.uint8)

        # Threshold the image to obtain a binary mask of the pale orange regions
        mask = cv2.inRange(hsv, lower_orange, upper_orange)

        # Apply morphological operations to clean up the mask
        kernel = np.ones((5, 5), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

        # Find contours in the mask
        contours, hierarchy = cv2.findContours(
            mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        img2 = cv2.drawContours(image, contours, -1, (255, 0, 0), 2)

        selected_idx = -1
        # Draw contours around the detected blobs on the original image
        for idx, contour in enumerate(contours):
            area = cv2.contourArea(contour)
            if area > 100:  # Filter out small contours
                (x, y, w, h) = cv2.boundingRect(contour)
                radius = int((w + h) / 4)
                selected_idx = idx
                cv2.circle(
                    image,
                    (int(x + w / 2), int(y + h / 2)),
                    int((w + h) / 4),
                    (0, 255, 0),
                    2,
                )

        if selected_idx == -1:
            rospy.loginfo("Cannot find orange blob for grasping probe")
            return 0.0

        # find the center of the circle
        M = cv2.moments(contours[selected_idx])
        cX = int(M["m10"] / M["m00"])
        cY = int(M["m01"] / M["m00"])

        # plot the center of the circle
        cv2.circle(image, (cX, cY), 7, (255, 255, 255), -1)
        cv2.putText(
            image,
            "center",
            (cX - 20, cY - 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 255, 255),
            2,
        )

        # Convert the image to grayscale
        gray = cv2.cvtColor(or_image, cv2.COLOR_BGR2GRAY)

        # Apply GaussianBlur to reduce noise
        gray_blur = cv2.GaussianBlur(gray, (5, 5), 0)

        ret, mask = cv2.threshold(gray_blur, 50, 255, cv2.THRESH_BINARY)
        # flip white and black, so that the cable is in white
        mask = cv2.bitwise_not(mask)

        # Apply morphological operations to clean up the mask
        kernel = np.ones((5, 5), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        cv2.imshow("mask", mask)

        # Find contours in the mask
        contours, hierarchy = cv2.findContours(
            mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        # remove all circles from the contours
        contours = [c for c in contours if cv2.arcLength(c, True) > 100]

        imx = cv2.drawContours(image, contours, -1, (255, 0, 0), 2)

        filtered_contours = []
        # check if the contour points are inside the circle and remove the contour if only all points are outside
        for idx, contour in enumerate(contours):
            contour_points = contour[:, 0, :]
            circle_center = np.array([[cX, cY]])
            dist = cdist(contour_points, circle_center)
            if all(dist > radius):
                # ignore this contour since it is completely outside the circle
                pass
            else:
                filtered_contours.append(contour)
        contours = filtered_contours

        selected_idx = -1
        # Draw contours around the detected cables on the original image
        for idx, contour in enumerate(contours):
            area = cv2.contourArea(contour)
            if area > 100:  # Filter out small contours
                selected_idx = idx
                (x, y, w, h) = cv2.boundingRect(contour)
                cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)

        if selected_idx == -1:
            rospy.loginfo("Cannot find cable for grasping probe")
            return 0.0

        M = cv2.moments(contours[selected_idx])
        bX = int(M["m10"] / M["m00"])
        bY = int(M["m01"] / M["m00"])

        # plot the direction of the cable with an arrow
        cv2.arrowedLine(image, (cX, cY), (bX, bY), (255, 0, 0), 2)

        # urint the angle of the cable direction wrt center
        angle = math.atan2(bY - cY, bX - cX) * 180.0 / math.pi

        # plot the angle of the cable direction wrt center
        cv2.putText(
            image,
            "angle: {:.2f} degrees".format(angle),
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 0, 255),
            2,
        )

        return angle

    def get_door_knob_error(self, save_debug_images=False):
        if save_debug_images:
            self.save_debug_image()

        """
        Get error for aligning with door knob using visual servoing
        """
        min_x = 280
        max_x = 1000
        min_y = 400
        max_y = 720
        # at height 0.11
        target_x = 362
        target_y = 280
        image = self.image[min_y:max_y, min_x:max_x]

        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (3, 3), 0)
        gray = cv2.Canny(gray, 80, 200)
        gray = dilate(gray)
        gray = erode(gray)
        circles = detect_door_circle(gray)
        if circles is not None:
            if len(circles) > 1:
                return None, None
            elif len(circles) == 1:
                cc = circles[0]
                cx, cy, r = cc[0], cc[1], cc[2]
                # Draw the circumference of the circle.
                cv2.circle(image, (cx, cy), r, (0, 255, 0), 2)
                error_x = target_x - cx
                error_y = target_y - cy
        else:
            error_x = None
            error_y = None
        # cv2.imshow("image", image)
        self.visual_servo_debug_img.publish(
            self.bridge.cv2_to_imgmsg(image, "bgr8")
        )

        # cv2.waitKey(1)
        return error_x, error_y

    def model_params(self):
        self.model.conf = 0.25  # NMS confidence threshold
        self.model.iou = 0.45  # NMS IoU threshold
        self.model.agnostic = False  # NMS class-agnostic
        self.model.multi_label = False  # NMS multiple labels per box
        self.model.max_det = 1000  # maximum number of detections per image

    def get_door_knob_error_2(self, save_debug_images=False):
        if save_debug_images:
            self.save_debug_image()

        image_copy = self.image.copy()

        target_x = 362
        target_y = 280

        results = self.model(self.image, size=640)  # try 480, 512, 640

        # handle the error with try and except
        try:
            if results.pred[0] is not None:  # if there are any detections
                predictions = results.pred[0]
                if predictions[:, 4]:
                    scores = predictions[:, 4]

                    # if more than one detection then take the one with highest score
                    max_score_index = scores.argmax()
                    max_score = scores[max_score_index]

                    # get the bounding box of the detection with highest score
                    # convert it to numpy array
                    boxes = predictions[max_score_index:, :4].asnumpy()

                    # TODO: add some conditions to avoid wrong detections, eg. like the area of the bounding box
                    boxes = predictions[:, :4]  # x1, y1, x2, y2

                    # find the center of the image
                    center = (image_copy.shape[1] / 2, image_copy.shape[0] / 2)

                    # draw vertical line at the center of the image
                    cv2.line(
                        image_copy,
                        (int(center[0]), 0),
                        (int(center[0]), image_copy.shape[0]),
                        (0, 0, 255),
                        1,
                    )

                    # find the center of the bounding box
                    center_box = (boxes[0][0] + boxes[0][2]) / 2, (
                        boxes[0][1] + boxes[0][3]
                    ) / 2

                    # show the center of the bounding box on the image
                    cv2.circle(
                        image_copy,
                        (int(center_box[0]), int(center_box[1])),
                        4,
                        (255, 255, 0),
                        1,
                    )

                    # find the error
                    error_x = target_x - center_box[0]
                    error_y = target_y - center_box[1]

                    # print the error on the image on the top left corner of the image
                    cv2.putText(
                        image_copy,
                        "error_x: {:.2f} error_y: {:.2f}".format(
                            error_x, error_y
                        ),
                        (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6,
                        (255, 255, 0),
                        2,
                    )

                    # # draw a horizontal line from the centroid to the target point (x-axis only)
                    horizontal_line = [
                        (center_box[0], center_box[1]),
                        (target_x, center_box[1]),
                    ]
                    cv2.line(
                        image_copy,
                        horizontal_line[0],
                        horizontal_line[1],
                        (0, 255, 0),
                        2,
                    )

                    # draw a vertical line from the end of the horizontal line to the target point (y-axis only)
                    vertical_line = [
                        (target_x, center_box[1]),
                        (target_x, target_y),
                    ]
                    cv2.line(
                        image_copy,
                        vertical_line[0],
                        vertical_line[1],
                        (0, 255, 0),
                        2,
                    )

                    # publish the debug image
                    self.img_pub.publish(
                        self.bridge.cv2_to_imgmsg(image_copy, "bgr8")
                    )

                    return error_x, error_y
            else:
                print("No predictions")
                return None, None
        except:
            print("No predictions")
            return None, None

    def get_orange_mask(self, img):
        lower = np.array([91, 120, 140])
        upper = np.array([111, 255, 255])
        mask = cv2.inRange(img, lower, upper)
        mask = dilate(mask)
        mask = erode(mask)
        return mask

    def get_probe_point_error(self, save_debug_images=False):
        if save_debug_images:
            self.save_debug_image()

        """
        Get error for circuit probe point for visual servoing
        """
        ## set at height of 0.3 m (i.e tool_pose_z = 0.3)
        ## if the probe ends up too far to the bottom, increase target_y
        ## if the probe ends up too far to the right, increase target_x
        target_x = 640
        target_y = 495
        if self.image is None:
            return None, None
        (height, width, c) = self.image.shape
        start_y = int(height / 4)
        start_x = int(width * 0.4)
        img = self.image[
            start_y : int(height - height / 4),
            start_x : int(width - width / 4),
            :,
        ]
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        mask = self.get_orange_mask(hsv)
        cx, cy = get_uppermost_contour(mask)
        if cx is None:
            return None, None
        cv2.circle(img, (cx, cy), 5, (0, 255, 0), 2)
        self.visual_servo_debug_img.publish(
            self.bridge.cv2_to_imgmsg(img, "bgr8")
        )
        error_x = target_x - (cx + start_x)
        error_y = target_y - (cy + start_y)
        return error_x, error_y

    def run_visual_servoing(
        self, vs_target_fn, target_height, save_debug_images=False
    ):
        success = False
        rospy.loginfo("Moving to correct height")
        while not rospy.is_shutdown():
            # we need to be at 0.3 height to do VS
            height_error = self.current_height - target_height
            msg = kortex_driver.msg.TwistCommand()
            msg.reference_frame = (
                kortex_driver.msg.CartesianReferenceFrame.CARTESIAN_REFERENCE_FRAME_TOOL
            )
            if height_error > 0.001:
                msg.twist.linear_z = 0.02
            elif height_error < -0.001:
                msg.twist.linear_z = -0.02
            else:
                break
            self.cart_vel_pub.publish(msg)
            self.loop_rate.sleep()

        rospy.loginfo("visual servoing")
        stop = False
        while not rospy.is_shutdown():
            if len(self.current_force_z) < 20:
                self.loop_rate.sleep()
                continue
            msg = kortex_driver.msg.TwistCommand()
            msg.reference_frame = (
                kortex_driver.msg.CartesianReferenceFrame.CARTESIAN_REFERENCE_FRAME_TOOL
            )
            error_x, error_y = vs_target_fn(save_debug_images)
            if error_x is None:
                msg.twist.linear_x = 0.0
            else:
                if error_x < 0:
                    msg.twist.linear_x = -0.005
                if error_x > 0:
                    msg.twist.linear_x = 0.005
                if abs(error_x) < 3:
                    msg.twist.linear_x = 0.0
                elif abs(error_x) < 10:
                    msg.twist.linear_x *= 0.5

                if error_y < 0:
                    msg.twist.linear_y = -0.005
                if error_y > 0:
                    msg.twist.linear_y = 0.005
                if abs(error_y) < 3:
                    msg.twist.linear_y = 0.0
                elif abs(error_y) < 10:
                    msg.twist.linear_y *= 0.5
            self.cart_vel_pub.publish(msg)
            if (
                msg.twist.linear_x == 0.0
                and msg.twist.linear_y == 0
                and error_x is not None
            ):
                break
            self.loop_rate.sleep()
        msg = kortex_driver.msg.TwistCommand()
        msg.reference_frame = (
            kortex_driver.msg.CartesianReferenceFrame.CARTESIAN_REFERENCE_FRAME_MIXED
        )
        msg.twist.linear_x = 0.0
        msg.twist.linear_y = 0.0
        self.cart_vel_pub.publish(msg)
        success = True
        return success

    def move_down_and_probe(self):
        """
        Once we have aligned with the probe circuit point, go down
        """
        #### Go down fast
        rospy.loginfo("moving down fast to probe circuit")
        self.current_force_z = []
        stop = False
        while not rospy.is_shutdown():
            if len(self.current_force_z) < 20:
                self.loop_rate.sleep()
                continue
            msg = kortex_driver.msg.TwistCommand()
            msg.reference_frame = (
                kortex_driver.msg.CartesianReferenceFrame.CARTESIAN_REFERENCE_FRAME_TOOL
            )
            msg.twist.linear_z = 0.05
            if (
                abs(np.mean(self.current_force_z) - self.current_force_z[-1])
                > 5.0
            ):
                rospy.loginfo("force threshold during fast motion")
                stop = True
                msg.twist.linear_z = 0.0
            if self.current_height < 0.21:  # just above the hole
                rospy.loginfo("height threshold during fast motion")
                stop = True
                msg.twist.linear_z = 0.0
            self.cart_vel_pub.publish(msg)
            if stop:
                break
            self.loop_rate.sleep()

        #### Go down slowly
        rospy.loginfo("moving down slowly to probe circuit")
        self.current_force_z = []
        stop = False
        while not rospy.is_shutdown():
            if len(self.current_force_z) < 20:
                self.loop_rate.sleep()
                continue
            msg = kortex_driver.msg.TwistCommand()
            msg.reference_frame = (
                kortex_driver.msg.CartesianReferenceFrame.CARTESIAN_REFERENCE_FRAME_TOOL
            )
            msg.twist.linear_z = 0.005
            if (
                abs(np.mean(self.current_force_z) - self.current_force_z[-1])
                > 5.0
            ):
                rospy.loginfo(
                    "Force difference threshold reached for probing circuit"
                )
                stop = True
                msg.twist.linear_z = 0.0
            if self.current_height < 0.185:  # already inside
                stop = True
                msg.twist.linear_z = 0.0
            self.cart_vel_pub.publish(msg)
            if stop:
                break
            self.loop_rate.sleep()
        probed = False
        if self.current_height < 0.188:
            probed = True

        #### Stop and go up
        msg = kortex_driver.msg.TwistCommand()
        msg.reference_frame = (
            kortex_driver.msg.CartesianReferenceFrame.CARTESIAN_REFERENCE_FRAME_MIXED
        )
        msg.twist.linear_z = 0.0
        self.cart_vel_pub.publish(msg)
        self.loop_rate.sleep()

        current_pose = self.arm.get_current_pose()
        current_pose.z += 0.07
        self.arm.send_cartesian_pose(current_pose)

        return probed

    def get_kinova_pose(self, pose_name):
        pose = rospy.get_param("~" + pose_name)

        sample = PoseStamped()
        sample.header.stamp = rospy.Time(0)
        sample.header.frame_id = "board_link"
        sample.pose.position.x = pose["pose"]["position"]["x"]
        sample.pose.position.y = pose["pose"]["position"]["y"]
        sample.pose.position.z = pose["pose"]["position"]["z"]
        sample.pose.orientation.x = pose["pose"]["orientation"]["x"]
        sample.pose.orientation.y = pose["pose"]["orientation"]["y"]
        sample.pose.orientation.z = pose["pose"]["orientation"]["z"]
        sample.pose.orientation.w = pose["pose"]["orientation"]["w"]

        pose_in_base_link = self.transform_utils.transformed_pose_with_retries(
            sample, "base_link", 3
        )

        pose_in_kinova_pose = get_kinovapose_from_pose_stamped(
            pose_in_base_link
        )

        return pose_in_kinova_pose

    def get_trajactory_poses(self, num_poses):
        """
        get poses

        input: num_poses: number of poses to get
        return: list of poses
        """

        pose_list = []
        for i in range(num_poses):
            pose_name = "pose" + str(i + 1)
            pose = self.get_kinova_pose(pose_name)
            pose_list.append(pose)
        return pose_list

    def save_debug_image(self):
        # get the current date and time
        now = datetime.datetime.now()

        # check if the directory exists
        if not os.path.exists(self.save_debug_image_dir):
            os.makedirs(self.save_debug_image_dir)

        if self.image is not None:
            cv2.imwrite(
                os.path.join(
                    self.save_debug_image_dir,
                    "DoorKnob_debug_image_{}.png".format(now),
                ),
                self.image,
            )
