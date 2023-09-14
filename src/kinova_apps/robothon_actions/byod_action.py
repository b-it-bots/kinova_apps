#!/usr/bin/env python3
import tf
import rospy
import numpy as np
import math
import cv2
import os
import datetime

from kortex_driver.msg import TwistCommand, CartesianReferenceFrame

from kinova_apps.abstract_action import AbstractAction
from kinova_apps.full_arm_movement import FullArmMovement
from geometry_msgs.msg import PoseStamped, Quaternion, Twist
from utils.transform_utils import TransformUtils
from utils.kinova_pose import (
    KinovaPose,
    get_kinovapose_from_pose_stamped,
    get_kinovapose_from_list,
)
from utils.force_measure import ForceMeasurmement
from sensor_msgs.msg import Image
from std_msgs.msg import String
from cv_bridge import CvBridge, CvBridgeError
import pytesseract


class ByodAction(AbstractAction):
    # TODO: fix the [ERROR] [1682429626.398549]: Received ACTION_ABORT notification problem

    def __init__(self, arm: FullArmMovement, transform_utils: TransformUtils):
        super().__init__(arm, transform_utils)
        self.arm = arm
        self.fm = ForceMeasurmement()
        self.tf_utils = transform_utils
        self.listener = tf.TransformListener()
        self.bridge = CvBridge()

        self.multimeter_poses = rospy.get_param("~multimeter_poses")
        self.joint_angles = rospy.get_param("~joint_angles")
        self.byod_poses = rospy.get_param("~byod_poses")
        self.power_button_poses = rospy.get_param("~power_button_poses")

        self.cartesian_velocity_pub = rospy.Publisher(
            "/my_gen3/in/cartesian_velocity", TwistCommand, queue_size=1
        )
        self.img_sub = rospy.Subscriber(
            "/camera/color/image_raw", Image, self.image_cb
        )
        # same topic as other debug images
        self.img_pub = rospy.Publisher(
            "/visual_servoing_debug_img", Image, queue_size=10
        )
        self.multimeter_value_pub = rospy.Publisher(
            "/multimeter_value", String, queue_size=10
        )
        self.save_debug_image_dir = "/home/b-it-bots/temp/robothon/BYOD"
        print("BYOD Action Initialized")

    def pre_perceive(self) -> bool:
        print("in pre perceive")

        success = self.arm.execute_gripper_command(0.35)
        if not success:
            return False
        success = self.arm.execute_gripper_command(1.0)
        if not success:
            return False

        return True

    def act(self) -> bool:
        print("in act")

        rospy.loginfo(">> Executing BYOD Trajectory <<")
        success = self.get_poses_and_follow_trajactory()
        if not success:
            return False

        rospy.loginfo(">> Pressing Power Button <<")
        success = self.press_power_button(target_status="on")
        if not success:
            return False

        rospy.loginfo(">> Rotate the Dial <<")
        success = self.rotate_dial(target_status="on")
        if not success:
            return False

        rospy.loginfo(">> Reading multimeter screen <<")
        success = self.read_multimeter_screen()
        if not success:
            return False

        rospy.loginfo(">> Pressing Power Button <<")
        success = self.press_power_button(target_status="off")
        if not success:
            return False

        rospy.loginfo(">> Rotate the Dial <<")
        success = self.rotate_dial(target_status="off")
        if not success:
            return False

        return True

    def verify(self) -> bool:
        print("in verify")
        return True

    def image_cb(self, msg):
        # get the image from the message
        try:
            image = self.bridge.imgmsg_to_cv2(msg, "bgr8")
        except CvBridgeError as e:
            print(e)
        self.image = image

    def get_poses_and_follow_trajactory(self):
        success = True
        pose = self.byod_poses
        # pose_list = {key: [] for key in pose.keys()}
        pose_list = {}
        for pose_name, i in zip(pose.keys(), pose.values()):
            pose_list[pose_name] = get_kinovapose_from_list(list(i.values()))

        # Go byod_pose in joint angles
        success &= self.arm.send_joint_angles(
            self.joint_angles["byod_safe_pose"]
        )
        if not success:
            return False

        trajectory = []

        max_velocity = 0.15
        max_angular_velocity = 15

        success &= self.arm.execute_gripper_command(0.60)  # open gripper

        trajectory.append(pose_list["pose1"])
        trajectory.append(pose_list["pose2"])

        success &= self.arm.traverse_waypoints(
            trajectory,
            max_lin_vel=max_velocity,
            max_ang_vel=max_angular_velocity,
        )
        if not success:
            return False

        success = self.arm.execute_gripper_command(0.95)  # close gripper
        if not success:
            return False

        success &= self.arm.send_cartesian_pose(pose_list["pose3"])
        success &= self.arm.send_cartesian_pose(pose_list["pose4"])
        success &= self.insert_probe(pose_list["pose5"])
        if not success:
            return False

        success &= self.arm.execute_gripper_command(0.60)  # open gripper
        if not success:
            return False

        trajectory.clear()
        trajectory.append(pose_list["pose6"])
        trajectory.append(pose_list["pose7"])
        trajectory.append(pose_list["pose8"])
        success &= self.arm.traverse_waypoints(
            trajectory,
            max_lin_vel=max_velocity,
            max_ang_vel=max_angular_velocity,
        )
        if not success:
            return False

        success &= self.arm.execute_gripper_command(0.95)  # close gripper
        if not success:
            return False

        trajectory.clear()
        trajectory.append(pose_list["pose9"])
        trajectory.append(pose_list["pose10"])
        trajectory.append(pose_list["pose11"])
        trajectory.append(pose_list["pose12"])

        success &= self.arm.traverse_waypoints(
            trajectory,
            max_lin_vel=max_velocity,
            max_ang_vel=max_angular_velocity,
        )
        if not success:
            return False

        success &= self.insert_probe(pose_list["pose13"])
        success &= self.arm.execute_gripper_command(0.60)  # open gripper

        if not success:
            return False

        trajectory.clear()
        trajectory.append(pose_list["pose14"])
        trajectory.append(pose_list["pose15"])

        success &= self.arm.traverse_waypoints(
            trajectory,
            max_lin_vel=max_velocity,
            max_ang_vel=max_angular_velocity,
        )
        if not success:
            return False

        return True

    def press_power_button(self, target_status: str):
        """
        switch the button on/off
        Args:
            target_status (str): target status of the button, either "on" or "off"
        """

        success = True
        # switch on the device

        success = self.arm.execute_gripper_command(1.0)
        if not success:
            return False

        pose = get_kinovapose_from_list(
            list(
                self.power_button_poses[
                    "button_" + target_status + "_up"
                ].values()
            )
        )
        success = self.arm.send_cartesian_pose(pose)
        if not success:
            return False
        approach_pose = get_kinovapose_from_list(
            list(
                self.power_button_poses[
                    "button_" + target_status + "_down"
                ].values()
            )
        )
        approach_pose.z += 0.03
        success = self.arm.send_cartesian_pose(approach_pose, max_lin_vel=0.05)
        if not success:
            return False
        rospy.sleep(0.5)  # for the arm to stabilize
        success = self.arm.move_down_with_caution(
            force_threshold=[5, 5, 10],
            tool_z_thresh=0.080,
            velocity=0.01,
            retract=True,
            retract_dist=0.04,
        )  # neg because arm is moving in -y axis
        if not success:
            return False
        return True

    def insert_probe(self, pose):
        pose.z += 0.03

        success = self.arm.send_cartesian_pose(pose)
        if not success:
            return False
        rospy.sleep(0.5)  # for the arm to stabilize

        rospy.loginfo("Moving down with caution")
        success = self.arm.move_down_with_caution(
            force_threshold=[4, 4, 4],
            tool_z_thresh=0.060,
            velocity=-0.01,
            approach_axis="y",
            retract=False,
        )  # neg because arm is moving in -y axis
        if not success:
            return False
        rospy.loginfo(">> probe reached<<")
        success = self.arm.execute_gripper_command(0.60)
        if not success:
            return False
        rospy.loginfo(">>Opened Gripper<<")
        return True

    def rotate_dial(self, target_status: str):
        # Go byod_pose in joint angles
        success = self.arm.send_joint_angles(
            self.joint_angles["multimeter_pre_pose"]
        )
        if not success:
            return False
        rospy.loginfo(">> Safe Pose Reached <<")

        pose = self.multimeter_poses

        dial_align_pose = pose["dial_align_pose"]
        dial_align_pose = get_kinovapose_from_list(list(dial_align_pose))
        dial_align_pose.z -= 0.03
        success = self.arm.send_cartesian_pose(dial_align_pose)
        if not success:
            return False

        if target_status == "off":
            # rotate the dial with cartesian
            current_pose = self.arm.get_current_pose()
            current_pose.theta_z_deg -= 21
            success = self.arm.send_cartesian_pose(current_pose)
            if not success:
                return False

        # open gripper
        success = self.arm.execute_gripper_command(0.50)
        if not success:
            return False

        # go down
        rospy.sleep(0.5)  # for the arm to stabilize
        success = self.arm.move_down_with_caution(
            force_threshold=[3, 3, 4.5],
            tool_z_thresh=0.045,
            velocity=0.01,
            retract=True,
            retract_dist=0.006,
        )
        if not success:
            return False
        success = self.arm.stop_arm_velocity()
        if not success:
            return False

        rospy.loginfo(">> Dial reached <<")

        # hold the dial
        success = self.arm.execute_gripper_command(1.0)
        if not success:
            return False

        rospy.loginfo(">> Dial held <<")

        if target_status == "on":
            # rotate the dial with cartesian
            current_pose = self.arm.get_current_pose()
            current_pose.theta_z_deg -= 21
            success = self.arm.send_cartesian_pose(current_pose)
            if not success:
                return False
        elif target_status == "off":
            # rotate the dial with cartesian
            current_pose = self.arm.get_current_pose()
            current_pose.theta_z_deg += 21
            success = self.arm.send_cartesian_pose(current_pose)
            if not success:
                return False

        # release the dial
        success = self.arm.execute_gripper_command(0.60)
        if not success:
            return False

        rospy.loginfo(">> Dial released <<")

        # go up
        current_pose = self.arm.get_current_pose()
        current_pose.z += 0.05
        success = self.arm.send_cartesian_pose(current_pose)
        if not success:
            return False

        rospy.loginfo(">> Dial rotated successfully<<")

        return True

    def read_multimeter_screen(self):
        pose = self.multimeter_poses

        # press red button
        rospy.loginfo(">> Pressing red button <<")
        red_button_pose = pose["red_button_press"]
        red_button_pose = get_kinovapose_from_list(list(red_button_pose))
        red_button_pose.z += 0.01
        success = self.arm.send_cartesian_pose(red_button_pose)
        if not success:
            return False

            # close gripper 60%
        success = self.arm.execute_gripper_command(0.60)
        if not success:
            return False

            # approach and press button
        rospy.sleep(0.5)  # for the arm to stabilize
        success = self.arm.move_down_with_caution(
            force_threshold=[4.0, 4.0, 12.0],
            tool_z_thresh=0.040,
            velocity=0.008,
            retract=True,
            retract_dist=0.015,
        )
        if not success:
            return False

        success = self.arm.stop_arm_velocity()
        if not success:
            return False
        rospy.loginfo(">> Red button pressed <<")

        # press white button
        rospy.loginfo(">> Pressing white button <<")
        white_button_pose = pose["white_button_press"]
        white_button_pose = get_kinovapose_from_list(list(white_button_pose))
        white_button_pose.z += 0.01
        success = self.arm.send_cartesian_pose(white_button_pose)
        if not success:
            return False

            # close gripper 30%
        success = self.arm.execute_gripper_command(0.60)
        if not success:
            return False

            # approach and press button
        rospy.sleep(0.5)  # for the arm to stabilize
        success = self.arm.move_down_with_caution(
            force_threshold=[4.0, 4.0, 12.0],
            tool_z_thresh=0.045,
            velocity=0.008,
            retract=True,
            retract_dist=0.015,
        )
        if not success:
            return False

        success = self.arm.stop_arm_velocity()
        if not success:
            return False
        rospy.loginfo(">> White button pressed <<")

        # Read multimeter screen
        screen_read_pose = pose["screen_read_pose"]
        screen_read_pose = get_kinovapose_from_list(list(screen_read_pose))
        success = self.arm.send_cartesian_pose(screen_read_pose)
        if not success:
            return False

        rospy.loginfo(">> Reading multimeter screen <<")

        # read the screen and publish the value
        readings = self.multimeter_screen_ocr(save_debug_images=True)

        # publish the value
        if readings:
            self.multimeter_value_pub.publish(readings)
            rospy.loginfo(">> Multimeter value published <<")
            rospy.loginfo(readings)
        else:
            rospy.loginfo(">> Trying to read multimeter screen again <<")

        return True

    def multimeter_screen_ocr(self, save_debug_images=False):
        if self.save_debug_images:
            self.save_debug_images()

        image_original = self.image.copy()

        # flip the image vertically
        img = cv2.flip(image_original, 0)
        # flip the image horizontally
        flipped_image = cv2.flip(img, 1)

        # ROI crop parameters
        min_x = 520
        max_x = 756
        min_y = 470
        max_y = 580

        # # crop to ROI
        image = self.image[min_y:max_y, min_x:max_x]

        # flip the image vertically
        image = cv2.flip(image, 0)
        # flip the image horizontally
        image = cv2.flip(image, 1)

        # rotate the image little but anti-clockwise to make the numbers straight
        image = self.rotate_image(image, 1)  # angle in degrees
        image_original = image.copy()

        # Convert image to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # blur the image
        blur = cv2.GaussianBlur(gray, (5, 5), 0)

        # Apply threshold to get image with only black and white
        thresh = cv2.threshold(
            blur, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU
        )[1]

        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=1)
        invert = 255 - opening

        # add padding to the image
        invert = cv2.copyMakeBorder(
            invert, 300, 300, 300, 300, cv2.BORDER_CONSTANT, value=255
        )

        # resize to 640x480
        invert = cv2.resize(invert, (640, 480))

        # erode the image and dilate the image
        kernel = np.ones((3, 3), np.uint8)
        invert = cv2.erode(invert, kernel, iterations=1)
        invert = cv2.dilate(invert, kernel, iterations=1)

        # config = ("-l eng --oem 3 --psm 6 -c tessedit_char_whitelist=0123456789")
        config = "-l ssd -c tessedit_char_whitelist=0123456789"
        result = pytesseract.image_to_string(invert, config=config)

        # only keep the numbers
        result = "".join([c for c in result if c.isdigit() or c == "."])

        # print the result on the image
        cv2.putText(
            flipped_image,
            str(result),
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 0, 255),
            2,
        )

        # publish the image
        self.img_pub.publish(self.bridge.cv2_to_imgmsg(invert, "mono8"))

        return str(result)

    def rotate_image(self, image, angle):
        # grab the dimensions of the image and then determine the
        # center
        (h, w) = image.shape[:2]
        (cX, cY) = ((w - 1) / 2.0, (h - 1) / 2.0)

        # grab the rotation matrix (applying the negative of the
        # angle to rotate clockwise), then grab the sine and cosine
        # (i.e., the rotation components of the matrix)

        # angle = -angle
        M = cv2.getRotationMatrix2D((cX, cY), angle, 1.0)  # angle in degrees

        # perform the actual rotation and return the image
        return cv2.warpAffine(image, M, (w, h))

    def save_debug_images(self):
        # get the current date and time
        now = datetime.datetime.now()

        # check if the directory exists
        if not os.path.exists(self.save_debug_image_dir):
            os.makedirs(self.save_debug_image_dir)

        if self.image is not None:
            cv2.imwrite(
                os.path.join(
                    self.save_debug_image_dir,
                    "BYOD_debug_image_{}.png".format(now),
                ),
                self.image,
            )
