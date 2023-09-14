#!/usr/bin/env python3

import rospy
from kinova_apps.abstract_action import AbstractAction
from kinova_apps.full_arm_movement import FullArmMovement
from utils.transform_utils import TransformUtils
from utils.kinova_pose import (
    get_kinovapose_from_list,
    get_kinovapose_from_pose_stamped,
)
from geometry_msgs.msg import PoseStamped, Quaternion
from kortex_driver.srv import *
from kortex_driver.msg import *
import tf
import os
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError
import cv2
import numpy as np
import math
import datetime
import pdb
import yolov5


class WindCableAction(AbstractAction):
    def __init__(
        self, arm: FullArmMovement, transform_utils: TransformUtils
    ) -> None:
        super().__init__(arm, transform_utils)
        self.debug = rospy.get_param("~debug", False)
        self.img_sub = rospy.Subscriber(
            "/camera/color/image_raw", Image, self.image_cb
        )
        self.img_pub = rospy.Publisher(
            "/visual_servoing_debug_img", Image, queue_size=10
        )
        self.image = None
        self.loop_rate = rospy.Rate(10)
        self.bridge = CvBridge()
        self.cart_vel_pub = rospy.Publisher(
            "/my_gen3/in/cartesian_velocity",
            kortex_driver.msg.TwistCommand,
            queue_size=1,
        )
        self.model = yolov5.load(
            "/home/b-it-bots/robothon_ros_workspace/src/kinova_apps/models/probe_holder_horizontal/probe_holder_horizontal_nano_ver2.pt"
        )
        self.model_params()
        self.save_debug_image_dir = "/home/b-it-bots/temp/robothon/windCable"

    def pre_perceive(self) -> bool:
        print("in pre perceive")

        # pre-perceive pose

        self.arm.execute_gripper_command(0.35)

        # get the pre-perceive pose from tf
        # msg = PoseStamped()
        # msg.header.frame_id = "wind_cable_link"
        # msg.header.stamp = rospy.Time(0)
        # wind_cable_pose = self.transform_utils.transformed_pose_with_retries(msg, "base_link", execute_arm=True, offset=[0, 0, math.pi/2])

        # # convert to kinova pose
        # kp = get_kinovapose_from_pose_stamped(wind_cable_pose)

        # kp.z += 0.05

        # # send to arm
        # rospy.loginfo("Sending pre-perceive pose to arm")
        # if not self.arm.send_cartesian_pose(kp):
        #     rospy.logerr("Failed to send pre-perceive pose to arm")
        #     return False

        return True

    def act(self) -> bool:
        # start visual servoing
        # rospy.loginfo("Starting visual servoing")
        # # TODO: fix the visual servoing
        # success = self.run_visual_servoing(self.detect_wind_cable, True, error_thresholds = [40, 50])

        # kp = self.arm.get_current_pose()

        # kp.z = 0.03

        # if not self.arm.send_cartesian_pose(kp):
        #     return False

        # self.arm.execute_gripper_command(1.0)

        # if not success:
        #     return False

        # wind cable
        rospy.loginfo("[wind action] starting winding")
        success = self.wind_cable()

        if not success:
            return False

        pose_for_tucking_kp = self.find_and_save_tucking_pose()

        # pick probe from holder
        success = self.pick_probe_from_holder()

        if not success:
            return False

        # tuck the probe into board
        rospy.loginfo("Tucking probe into board")
        success = self.tuck_probe_into_board(pose_for_tucking_kp)

        return success

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

    def get_pose_in_board(self, pose: list):
        msg = PoseStamped()
        msg.header.frame_id = "board_link"
        msg.pose.position.x = pose[0]
        msg.pose.position.y = pose[1]
        msg.pose.position.z = pose[2]
        msg.pose.orientation.x = pose[3]
        msg.pose.orientation.y = pose[4]
        msg.pose.orientation.z = pose[5]
        msg.pose.orientation.w = pose[6]

        # convert to base_link frame
        msg_in_base = self.transform_utils.transformed_pose_with_retries(
            msg, "base_link"
        )

        # convert to kinova_pose
        kp = get_kinovapose_from_pose_stamped(msg_in_base)

        return kp

    def wind_cable(self) -> bool:
        # go and pick at board
        self.arm.execute_gripper_command(0.6)

        p1 = rospy.get_param("~wind_poses/p1")
        p2 = rospy.get_param("~wind_poses/p2")

        traj1 = []
        traj1.append(self.get_pose_in_board(p1))
        traj1.append(self.get_pose_in_board(p2))

        if not self.arm.traverse_waypoints(traj1):
            return False

        self.arm.execute_gripper_command(0.9)

        # drag and drop
        p3 = rospy.get_param("~wind_poses/p3")

        p3_kp = self.get_pose_in_board(p3)

        if not self.arm.send_cartesian_pose(p3_kp):
            return False

        self.arm.execute_gripper_command(0.35)

        # pick again
        p4 = rospy.get_param("~wind_poses/p4")

        p4_kp = self.get_pose_in_board(p4)

        if not self.arm.send_cartesian_pose(p4_kp):
            return False

        self.arm.execute_gripper_command(0.955)

        # wind
        waypoints = []
        for i in range(5, 30):
            pose = rospy.get_param("~wind_poses/p" + str(i))

            kp = self.get_pose_in_board(pose)

            waypoints.append(kp)
            # if not self.arm.send_cartesian_pose(kp):
            #     return False

        if not self.arm.traverse_waypoints(waypoints):
            return False

        self.arm.execute_gripper_command(0.85)
        self.arm.execute_gripper_command(0.7)
        self.arm.execute_gripper_command(0.35)
        current_pose = self.arm.get_current_pose()
        current_pose.z += 0.1
        self.arm.send_cartesian_pose(current_pose)
        return True

    def pick_probe_from_holder(self):
        # # go to the probe pick perceive position above the holder
        self.arm.execute_gripper_command(0.35)

        current_pose = self.arm.get_current_pose()
        current_pose.z += 0.05
        self.arm.send_cartesian_pose(current_pose)

        perceive_board_pose = rospy.get_param(
            "~joint_angles/perceive_board_pose"
        )
        success = self.arm.send_joint_angles(perceive_board_pose)

        safe_pose_after_probe_placement = rospy.get_param(
            "~probe_action_poses/safe_pose_after_probe_placement"
        )
        safe_pose_after_probe_placement = get_kinovapose_from_list(
            safe_pose_after_probe_placement
        )
        rospy.loginfo(
            "[probe_action] moving to safe pose after probe placement"
        )
        success = self.arm.send_cartesian_pose(
            safe_pose_after_probe_placement, max_lin_vel=0.05
        )

        probe_place_pre_holder_pose = rospy.get_param(
            "~probe_action_poses/probe_place_pre_holder_pose"
        )
        probe_place_pre_holder_pose = get_kinovapose_from_list(
            probe_place_pre_holder_pose
        )
        rospy.loginfo("[probe_action] moving to probe place pre holder")
        success = self.arm.send_cartesian_pose(
            probe_place_pre_holder_pose, max_lin_vel=0.05
        )

        tucking_probe_holder_pick = rospy.get_param(
            "~probe_action_poses/tucking_probe_holder_pick"
        )
        tucking_probe_holder_pick = get_kinovapose_from_list(
            tucking_probe_holder_pick
        )
        rospy.loginfo("[probe_action] moving to probe place holder position")
        success = self.arm.send_cartesian_pose(
            tucking_probe_holder_pick, max_lin_vel=0.05
        )
        if not success:
            rospy.logerr(
                "[probe_action] Failed to move to the probe place holder position"
            )
            return False

        # Moving Down from the probe_place_in_holder_pose down
        # probe_holder_pick_for_tucking.x += 0.01
        # success = self.arm.send_cartesian_pose(probe_holder_pick_for_tucking, max_lin_vel=0.05)
        # if not success:
        #    rospy.logerr("[probe_action] Failed to move to the probe place holder position")
        #    return False
        # close the gripper
        self.arm.execute_gripper_command(1.0)

        # move up a bit
        tucking_probe_holder_pick.z += 0.1

        success = self.arm.send_cartesian_pose(
            tucking_probe_holder_pick, max_lin_vel=0.05
        )

        if not success:
            rospy.logerr("[probe_action] Failed to move up the probe")
            return False

        rospy.loginfo("[probe_action] probe picked")

        return True

    def tuck_probe_into_board(self, pose_for_tucking_kp) -> bool:
        rospy.loginfo("[probe_action] moving to probe initial position")
        success = self.arm.send_cartesian_pose(pose_for_tucking_kp)

        if not success:
            rospy.logerr(
                "[probe_action] Failed to move to the probe initial position"
            )
            return False

        rospy.loginfo("[probe_action] reached probe initial position")

        rospy.loginfo("[probe_action] moving away from holder")
        success = self.arm.move_with_velocity(0.03, 2.0, "y")
        if not success:
            rospy.logerr("[probe_action] Failed to move away from holder")
            return False

        current_pose = self.arm.get_current_pose()

        # use velocity control to move the probe down
        rospy.loginfo("[probe_action] moving down the probe")
        # success = self.arm.move_down_with_caution(force_threshold=[4,4,1.75], velocity=0.005, tool_z_thresh=0.10, retract_dist=0.008)

        pose_for_tucking_kp.z = 0.1173 + 0.00225
        pose_for_tucking_kp.x = current_pose.x
        pose_for_tucking_kp.y = current_pose.y

        success = self.arm.send_cartesian_pose(
            pose_for_tucking_kp, max_lin_vel=0.01
        )

        if not success:
            rospy.logerr("[probe_action] Failed to move down the probe")
            return False

        rospy.loginfo("[probe_action] moved down the probe")

        # move the probe back in x direction for 4cm
        # success = self.arm.move_with_velocity(-0.025, 3, 'y')

        # This moves the arm forward with some force for 1.5 cm .. Its 1.5 cm to just touch theprobe not full inserting
        self.arm.move_down_with_caution(
            approach_axis="y",
            time=8,
            distance=-0.015,
            force_threshold=[10, 10, 10],
        )

        if not success:
            rospy.logerr("Failed to move back the probe")
            return False

        self.arm.execute_gripper_command(0.6)  # open the gripper

        current_pose = self.arm.get_current_pose()
        current_pose.z += 0.04
        # move away from and above holder
        success = self.arm.send_cartesian_pose(current_pose)

        return True

    def run_visual_servoing(
        self, vs_target_fn, run=True, error_thresholds=[5, 10]
    ):
        stop = False
        while not rospy.is_shutdown():
            if self.image is None:
                rospy.loginfo("waiting for image")
                self.loop_rate.sleep()
                continue
            msg = kortex_driver.msg.TwistCommand()
            msg.reference_frame = (
                kortex_driver.msg.CartesianReferenceFrame.CARTESIAN_REFERENCE_FRAME_TOOL
            )
            x_error, y_error = vs_target_fn(True)
            if x_error is None:
                print("none")
                msg.twist.linear_x = 0.0
            if y_error is None:
                print("none")
                msg.twist.linear_y = 0.0
            if x_error is not None:
                rospy.loginfo("X Error: %.2f" % (x_error))
                if x_error < 0:
                    msg.twist.linear_x = -0.005
                if x_error > 0:
                    msg.twist.linear_x = 0.005
                # TODO : Make this separate for both
                if (
                    abs(x_error) < error_thresholds[0]
                ):  # TODO 40 for picking cable 5 for tcuking alignment
                    msg.twist.linear_x = 0.0
                elif (
                    abs(x_error) < error_thresholds[1]
                ):  # TODO 50 for picking cable
                    msg.twist.linear_x *= 0.5

            if y_error is not None:
                rospy.loginfo("Y Error: %.2f" % (y_error))
                if y_error < 0:
                    msg.twist.linear_y = -0.005
                if y_error > 0:
                    msg.twist.linear_y = 0.005
                if abs(y_error) < 3:
                    msg.twist.linear_y = 0.0
                elif abs(y_error) < 10:
                    msg.twist.linear_y *= 0.5
            if run:
                self.cart_vel_pub.publish(msg)
                if (
                    msg.twist.linear_x == 0.0
                    and msg.twist.linear_y == 0.0
                    and x_error is not None
                ):
                    break
            self.loop_rate.sleep()
        msg = kortex_driver.msg.TwistCommand()
        msg.reference_frame = (
            kortex_driver.msg.CartesianReferenceFrame.CARTESIAN_REFERENCE_FRAME_MIXED
        )
        self.cart_vel_pub.publish(msg)
        return True

    def detect_wind_cable(self, save_image=False):
        if save_image:
            self.save_debug_image()

        # parameters
        circularity_threshold_min = 0.0
        circularity_threshold_max = 0.5
        contours_area_threshold_min = 3000
        contours_area_threshold_max = 15000

        # image center coordinates
        image_center_x = self.image.shape[1] // 2
        image_center_y = self.image.shape[0] // 2

        # draw a rectangle on the image from the center of the image
        x_axis_right = 500
        x_axis_left = 300
        y_axis_top = 10
        y_axis_bottom = 200

        # crop the ROI from the image with the rectangle
        roi = self.image[
            image_center_y - y_axis_top : image_center_y + y_axis_bottom,
            image_center_x - x_axis_left : image_center_x + x_axis_right,
        ]

        cv2.rectangle(
            roi, (0, 0), (roi.shape[1], roi.shape[0]), (255, 255, 255), 20
        )

        roi_copy = roi.copy()
        roi_copy_2 = roi.copy()

        # convert the image to grayscale
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        # apply gaussian blur to the image
        blur = cv2.GaussianBlur(gray, (5, 5), 0)
        # otsu thresholding
        # ret, thresh = cv2.threshold(
        #     blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        ret, blur = cv2.threshold(blur, 100, 255, cv2.THRESH_BINARY)
        # apply canny edge detection
        canny = cv2.Canny(blur, 50, 150)
        # apply dilation
        kernel = np.ones((3, 3), np.uint8)
        canny = cv2.dilate(canny, kernel, iterations=1)
        # find the contours
        contours, _ = cv2.findContours(
            canny, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE
        )

        # draw the contours on the image
        cv2.drawContours(roi_copy, contours, -1, (0, 255, 0), 2)

        # filter out black contours
        filtered_contours = []
        for contour in contours:
            # Calculate area and perimeter of the contour
            area = cv2.contourArea(contour)
            perimeter = cv2.arcLength(contour, True)
            # filter out small contours
            if (
                area < contours_area_threshold_min
                or area > contours_area_threshold_max
            ):
                rospy.logwarn(
                    "Removing contour with area: %.2f, min:%.2f, max: %.2f"
                    % (
                        area,
                        contours_area_threshold_min,
                        contours_area_threshold_max,
                    )
                )
                continue

            # print("Area: {}".format(area))

            # Calculate circularity of the contour
            circularity = (4 * np.pi * area) / (perimeter**2)

            # print("Circularity: {}".format(circularity))

            if (
                circularity < circularity_threshold_min
                or circularity > circularity_threshold_max
            ):
                continue

            # draw contours on the image
            # cv2.drawContours(roi_copy_2, [contour], -1, (0, 255, 0), 2)
            # cv2.imshow("Contours", roi_copy_2)
            # cv2.waitKey(0)
            # cv2.destroyAllWindows()

            filtered_contours.append(contour)

        # print("Number of filtered contours: {}".format(len(filtered_contours)))

        # draw a horizontal line in the middle of the image
        horizontal_line = [
            (0, roi_copy_2.shape[0] // 2),
            (roi_copy_2.shape[1], roi_copy_2.shape[0] // 2),
        ]
        cv2.line(
            roi_copy_2, horizontal_line[0], horizontal_line[1], (0, 0, 255), 2
        )

        # draw a vertical line in the middle of the image
        vertical_line = [
            (roi_copy_2.shape[1] // 2, 0),
            (roi_copy_2.shape[1] // 2, roi_copy_2.shape[0]),
        ]
        cv2.line(roi_copy_2, vertical_line[0], vertical_line[1], (0, 0, 255), 2)

        # NOTE: it should only be one contour
        if len(filtered_contours) > 1:
            print("More than one contour found!")
            print("TODO 1: failure recovery mechanism")
            return (None, None)

        elif len(filtered_contours) == 1:
            # get the contour points
            contour_points = filtered_contours[0]

            # get the bottom most point of the contour
            bottom_most_point = contour_points[
                contour_points[:, :, 1].argmax()
            ][0]

            # draw a circle on the bottom most point
            cv2.circle(
                roi_copy_2,
                (bottom_most_point[0], bottom_most_point[1]),
                5,
                (0, 0, 255),
                -1,
            )

            error = (roi.shape[1] // 2) - bottom_most_point[0]
            print("Error: {}".format(error))

            # print the error on the image
            cv2.putText(
                roi_copy_2,
                "Error: {}".format(error),
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 0, 255),
                2,
            )

            # draw the error line on the image from the centroid to the vertical line
            error_line = [
                bottom_most_point,
                (roi.shape[1] // 2, bottom_most_point[1]),
            ]
            cv2.line(roi_copy_2, error_line[0], error_line[1], (255, 0, 0), 2)

            # publish the debug image
            self.img_pub.publish(self.bridge.cv2_to_imgmsg(roi_copy_2, "bgr8"))

            return (error, None)

        else:
            print("No contour found!")
            return (None, None)

    def find_and_save_tucking_pose(self):
        # current_pose = self.arm.get_current_pose()
        # current_pose.z = 0.11
        # self.arm.send_cartesian_pose(current_pose)
        probe_initial_pose_kp = self.transform_utils.transform_pose_frame_name(
            reference_frame_name="probe_initial_link",
            target_frame_name="base_link",
            offset_linear=[0.0, 0.00, 0.08],
            offset_rotation_euler=[math.pi, 0.0, math.pi / 2],
        )

        # send the probe initial position to the arm
        rospy.loginfo("[probe_action] moving to probe initial position")
        success = self.arm.send_cartesian_pose(probe_initial_pose_kp)

        if not success:
            rospy.logerr(
                "[probe_action] Failed to move to the probe initial position"
            )
            return False

        # visual servo to find correct location to place probe before picking the probe
        success = self.run_visual_servoing(
            self.detect_probe_holder_horizontal, True, error_thresholds=[5, 10]
        )
        pose_for_tucking = self.arm.get_current_pose()
        if pose_for_tucking.theta_z_deg > 0:
            pose_for_tucking.theta_z_deg -= 180.0
        else:
            pose_for_tucking.theta_z_deg += 180.0
        return pose_for_tucking

    def model_params(self):
        self.model.conf = 0.25  # NMS confidence threshold
        self.model.iou = 0.45  # NMS IoU threshold
        self.model.agnostic = False  # NMS class-agnostic
        self.model.multi_label = False  # NMS multiple labels per box
        self.model.max_det = 1000  # maximum number of detections per image

    def detect_probe_holder_horizontal(self, save_image=False):
        if save_image:
            self.save_debug_image()

        image_copy = self.image.copy()

        results = self.model(self.image, size=640)  # try 480, 512, 640

        # handle the error with try and except
        try:
            if results.pred[0] is not None:  # if there are any detections
                predictions = results.pred[0]
                if predictions[:, 4]:
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

                    # TODO: Check if the probe can be picked near the tip
                    # find the error in x direction
                    error_x = (
                        center[0] - center_box[0] + 27.0
                    )  # magic number for aligning the tip of probe to the center

                    # print the error on the image on the top left corner of the image
                    cv2.putText(
                        image_copy,
                        "Error: " + str(error_x.numpy()),
                        (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1,
                        (255, 255, 0),
                        2,
                    )

                    # draw the error line from the center of bounding box to the y axis of the image
                    cv2.line(
                        image_copy,
                        (int(center_box[0]), int(center_box[1])),
                        (int(center_box[0] + error_x), int(center_box[1])),
                        (0, 255, 0),
                        2,
                    )

                    # publish the debug image
                    self.img_pub.publish(
                        self.bridge.cv2_to_imgmsg(image_copy, "bgr8")
                    )

                return error_x.numpy(), None  # error in x direction
            else:
                print("No predictions")
                return None, None
        except:
            print("No predictions")
            return None, None

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
                    "WindCable_debug_image_{}.png".format(now),
                ),
                self.image,
            )
