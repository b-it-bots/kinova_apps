#!/usr/bin/env python3
#
import rospy
import numpy as np
import kortex_driver.msg
from typing import Type, TypeVar
from kinova_apps.full_arm_movement import FullArmMovement
from kinova_apps.abstract_action import AbstractAction
from utils.transform_utils import TransformUtils
from cv_bridge import CvBridge, CvBridgeError
import cv2
from sensor_msgs.msg import Image
import math
import scipy.spatial as spatial
import datetime
import os


def dilate(img, dilation_size=1):
    dilation_shape = cv2.MORPH_RECT
    element = cv2.getStructuringElement(
        dilation_shape,
        (2 * dilation_size + 1, 2 * dilation_size + 1),
        (dilation_size, dilation_size),
    )
    img = cv2.dilate(img, element)
    return img


def erode(img, erosion_size=1):
    erosion_shape = cv2.MORPH_RECT
    element = cv2.getStructuringElement(
        erosion_shape,
        (2 * erosion_size + 1, 2 * erosion_size + 1),
        (erosion_size, erosion_size),
    )
    img = cv2.erode(img, element)
    return img


class PlugRemoveSlidAction(AbstractAction):
    """
    Assumption: We have approximate pose of the plug .

    - Move arm to the top of the plugs
    - Do visual servoing and adjust arm
    - Open gripper
    - Move arm with velocity control downwards
    - stope when the force is higher
    - retreate above
    - close gripper
    - move up
    - go to fixed insert position
    - Move downwards with velocity control
    - Check force for 2 levels - (board collision and slid force)  needs to be identified
    - Stop if board collision is hit
    - retreat above and random sample x and y pose
    - if slid force is there for some amount of time then collision force comes then open gripper
    -
    """

    def __init__(
        self, arm: FullArmMovement, transform_utils: TransformUtils
    ) -> None:
        super(PlugRemoveSlidAction, self).__init__(arm, transform_utils)
        self.current_force_z = []
        self.current_height = None
        self.base_feedback_sub = rospy.Subscriber(
            "/my_gen3/base_feedback",
            kortex_driver.msg.BaseCyclic_Feedback,
            self.base_feedback_cb,
        )
        self.cart_vel_pub = rospy.Publisher(
            "/my_gen3/in/cartesian_velocity",
            kortex_driver.msg.TwistCommand,
            queue_size=1,
        )
        self.img_sub = rospy.Subscriber(
            "/camera/color/image_raw", Image, self.image_cb
        )
        self.img_pub = rospy.Publisher(
            "/visual_servoing_debug_img", Image, queue_size=10
        )
        self.loop_rate = rospy.Rate(10.0)
        self.image = None
        self.bridge = CvBridge()
        self.move_up_done = False
        self.move_down_done = False
        self.close_gripper_done = False
        self.save_debug_images_dir = "/home/b-it-bots/temp/robothon"

    def base_feedback_cb(self, msg):
        self.current_force_z.append(msg.base.tool_external_wrench_force_z)
        if len(self.current_force_z) > 25:
            self.current_force_z.pop(0)
        self.current_height = msg.base.tool_pose_z

    def pre_perceive(self) -> bool:
        print("in pre perceive")

        pre_height_above_button = rospy.get_param(
            "~pre_height_above_button", 0.3
        )
        kinova_pose = self.transform_utils.transform_pose_frame_name(
            reference_frame_name="meter_plug_black_link",
            target_frame_name="base_link",
            offset_linear=[0.00, 0.00, pre_height_above_button],
            offset_rotation_euler=[math.pi, 0.0, math.pi / 2],
        )

        self.arm.execute_gripper_command(0.5)
        self.arm.send_cartesian_pose(kinova_pose)
        return True

    def act(self) -> bool:
        print("in act")
        # Allign camera
        self.run_visual_servoing(
            self.align_black_port_2, save_debug_images=True, run=True
        )
        current_pose = self.arm.get_current_pose()
        current_pose.z -= 0.03
        self.arm.send_cartesian_pose(current_pose)
        self.move_down_velocity_control()
        grasp_height = self.current_height
        self.arm.execute_gripper_command(1.0)  # close gripper
        self.move_up_velocity_control()
        self.move_forward()
        inserted = False
        retries = 0
        max_insert_retries = rospy.get_param("~max_insert_retries", 5)
        while not inserted:
            self.run_visual_servoing(
                self.align_red_port, save_debug_images=False, run=True
            )
            # open the gripper a bit to allow some compliance
            # self.arm.execute_gripper_command(0.9)
            inserted = self.move_down_insert(grasp_height)
            retries += 1
            if retries > max_insert_retries:
                break
        if not inserted:
            # if we haven't inserted the plug, release it
            self.arm.execute_gripper_command(0.3)
        return True

    def verify(self) -> bool:
        return True

    def image_cb(self, msg):
        # get the image from the message
        try:
            image = self.bridge.imgmsg_to_cv2(msg, "bgr8")
        except CvBridgeError as e:
            print(e)
        self.image = image

    def run_visual_servoing(
        self, vs_target_fn, save_debug_images=False, run=True
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
            x_error, y_error = vs_target_fn(save_debug_images)
            if x_error is None:
                msg.twist.linear_x = 0.0
            if y_error is None:
                msg.twist.linear_y = 0.0
            if x_error is not None:
                rospy.loginfo("X Error: %.2f" % (x_error))
                if x_error < 0:
                    msg.twist.linear_x = -0.005
                if x_error > 0:
                    msg.twist.linear_x = 0.005
                if abs(x_error) < 3:
                    msg.twist.linear_x = 0.0
                elif abs(x_error) < 10:
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
                    and msg.twist.linear_y == 0
                    and x_error is not None
                    and y_error is not None
                ):
                    break
            self.loop_rate.sleep()
        msg = kortex_driver.msg.TwistCommand()
        msg.reference_frame = (
            kortex_driver.msg.CartesianReferenceFrame.CARTESIAN_REFERENCE_FRAME_MIXED
        )
        self.cart_vel_pub.publish(msg)

    def align_black_port(self, save_debug_images=False):
        if save_debug_images:
            self.save_debug_images()

        # parameters
        circularity_threshold_low = 0.82
        contours_area_threshold_low = 3000
        contours_area_threshold_high = 9000
        red_color_threshold_low = 10
        red_color_threshold_high = 80

        # crop to ROI
        min_x = 280
        max_x = 1000
        min_y = 100
        max_y = 710
        ## These are targets when tool_pose_z = approx 0.148 m
        target_x = 356
        target_y = 330

        # crop to ROI
        image = self.image[min_y:max_y, min_x:max_x]
        image_copy = image.copy()

        # find the contours
        # convert the image to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(15, 15))
        gray = clahe.apply(gray)
        # apply gaussian blur to the image
        blur = cv2.GaussianBlur(gray, (5, 5), 0)
        # otsu thresholding
        ret, blur = cv2.threshold(
            blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
        )
        # apply canny edge detection
        canny = cv2.Canny(blur, 50, 150)
        # find the contours
        contours, _ = cv2.findContours(
            canny, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        # filter out black contours
        filtered_contours = []
        for contour in contours:
            # Calculate area and perimeter of the contour
            area = cv2.contourArea(contour)
            perimeter = cv2.arcLength(contour, True)

            # filter out small contours
            if (
                area < contours_area_threshold_low
                or area > contours_area_threshold_high
            ):
                continue

            # Calculate circularity of the contour (reference: https://en.wikipedia.org/wiki/Roundness)
            circularity = (4 * np.pi * area) / (perimeter**2)

            # filter out non circular contours
            if circularity < circularity_threshold_low:
                continue

            # draw the contour on the image
            # cv2.drawContours(image, [contour], -1, (0, 255, 0), 2)

            # create a mask of the contour
            n_mask = np.zeros(image.shape[:2], dtype="uint8")
            cv2.drawContours(n_mask, [contour], 0, (255, 255, 255), -1)

            # get the mean color of the contour using the mask
            mean_color = cv2.mean(image_copy, mask=n_mask)

            # filter out non red contours
            if (
                mean_color[0] < red_color_threshold_high
                and mean_color[0] > red_color_threshold_low
            ):
                filtered_contours.append(contour)

        rospy.loginfo_throttle(
            2, "Number of filtered contours: {}".format(len(filtered_contours))
        )

        # NOTE: it should only be one contour
        if len(filtered_contours) > 1:
            print("More than one contour found!")
            print("TODO 1: failure recovery mechanism")
            return None, None

        elif len(filtered_contours) == 1:
            # draw the contour on the image
            cv2.drawContours(image, filtered_contours, -1, (0, 255, 0), 2)

            # display the image with target points
            cv2.circle(image, (target_x, target_y), 5, (255, 255, 0), -1)

            # draw a horizontal line from the target point
            cv2.line(
                image, (0, target_y), (image.shape[1], target_y), (0, 0, 255), 2
            )
            # draw a vertical line from the target point
            cv2.line(
                image, (target_x, 0), (target_x, image.shape[0]), (0, 0, 255), 2
            )

            # calculate the centroid of the contour
            M = cv2.moments(filtered_contours[0])
            centroid_x = int(M["m10"] / M["m00"])
            centroid_y = int(M["m01"] / M["m00"])

            # draw the centroid on the image
            cv2.circle(image, (centroid_x, centroid_y), 5, (0, 0, 255), -1)

            # calculate the error
            error_x = target_x - centroid_x
            error_y = target_y - centroid_y
            rospy.loginfo(
                "Centroid: %d, %d, Error: %d, %d"
                % (centroid_x, centroid_y, error_x, error_y)
            )
            # print('Centroid: %d, %d, Error: %d, %d' % (centroid_x, centroid_y, error_x, error_y))

            # print the error in the image
            cv2.putText(
                image,
                "Error: {}, {}".format(error_x, error_y),
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 255, 0),
                2,
            )

            # draw a horizontal error line from the centroid to the target point (x-axis only)
            horizontal_line = [(centroid_x, centroid_y), (target_x, centroid_y)]
            cv2.line(
                image, horizontal_line[0], horizontal_line[1], (0, 255, 0), 2
            )

            # draw a vertical error line from the end of the horizontal line to the target point (y-axis only)
            vertical_line = [(target_x, centroid_y), (target_x, target_y)]
            cv2.line(image, vertical_line[0], vertical_line[1], (0, 255, 0), 2)

            # show the result
            # cv2.imshow("Filtered Contours", image)
            # cv2.waitKey(0)
            # cv2.destroyAllWindows()

        else:
            rospy.loginfo_throttle(2, "No contour found!")
            error_x = None
            error_y = None

        self.img_pub.publish(self.bridge.cv2_to_imgmsg(image, "bgr8"))
        return error_x, error_y

    def align_black_port_2(self, save_debug_images=False):
        if save_debug_images:
            self.save_debug_images()

        # parameters
        red_lower1 = [0, 50, 50]
        red_upper1 = [4, 255, 255]
        red_lower2 = [177, 50, 50]
        red_upper2 = [180, 255, 255]
        contours_area_threshold_low = 3000
        contours_area_threshold_high = 9000
        visualization_flag = False
        # ROI crop parameters
        min_x = 280
        max_x = 1000
        min_y = 100
        max_y = 710
        ## These are targets when tool_pose_z = approx 0.148 m
        target_x = 356
        target_y = 330

        # display the image
        # cv2.imshow("Input image", image)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()

        # crop to ROI
        image = self.image[min_y:max_y, min_x:max_x]
        image_original = image.copy()
        image_original_copy_2 = image.copy()

        if visualization_flag:
            # show the result
            cv2.imshow("Input ROI image", image)
            cv2.waitKey(0)
            cv2.destroyAllWindows()

        # convert to HSV color space
        image_hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

        # define the lower and upper boundaries of the "red" HSV pixels
        lower1 = np.array(red_lower1)
        upper1 = np.array(red_upper1)
        lower2 = np.array(red_lower2)
        upper2 = np.array(red_upper2)

        # create the mask
        mask1 = cv2.inRange(image_hsv, lower1, upper1)
        mask2 = cv2.inRange(image_hsv, lower2, upper2)
        mask = cv2.bitwise_or(mask1, mask2)

        if visualization_flag:
            # show the result
            cv2.imshow("Mask", mask)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
            pass

        # Apply morphological transformations to remove noise
        kernel = np.ones((5, 5), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

        if visualization_flag:
            # show the result
            cv2.imshow("Morphological Transformed Mask", mask)
            cv2.waitKey(0)
            cv2.destroyAllWindows()

        # find contours in the mask
        contours, hierarchy = cv2.findContours(
            mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        # no contours found
        if len(contours) == 0:
            rospy.loginfo("[plug removal] No contours found!")
            return None, None

        # filter out small contours
        filtered_contours = []
        for contour in contours:
            # Calculate area and perimeter of the contour
            area = cv2.contourArea(contour)
            perimeter = cv2.arcLength(contour, True)

            if (
                area < contours_area_threshold_low
                or area > contours_area_threshold_high
            ):
                continue

            if visualization_flag:
                print("Area: ", area)
                # draw the contour on the original image
                cv2.drawContours(
                    image_original_copy_2, [contour], -1, (0, 255, 0), 2
                )
                # show the result
                cv2.imshow("Filtered Contour", image_original_copy_2)
                cv2.waitKey(0)
                cv2.destroyAllWindows()

            filtered_contours.append(contour)

        # no contours found
        if len(filtered_contours) == 0:
            rospy.loginfo("[plug removal] No filtered contours found!")
            return None, None

        # make a mask of the drawn contours
        mask = np.zeros(image_original.shape[:2], dtype="uint8")
        cv2.drawContours(mask, filtered_contours, -1, 255, -1)

        if visualization_flag:
            # show the result
            cv2.imshow("Masked Contours", mask)
            cv2.waitKey(0)
            cv2.destroyAllWindows()

        # smooth the edges of the mask
        mask = cv2.GaussianBlur(mask, (5, 5), 0)

        if visualization_flag:
            # show the result
            cv2.imshow("Masked Contours blur", mask)
            cv2.waitKey(0)
            cv2.destroyAllWindows()

        # find hough circles
        circles = cv2.HoughCircles(
            mask,
            cv2.HOUGH_GRADIENT,
            1,
            20,
            param1=50,
            param2=10,
            minRadius=40,
            maxRadius=60,
        )

        if visualization_flag:
            # draw the circles
            if circles is not None:
                circles = np.uint16(np.around(circles))
                for centroid_x, centroid_y, r in circles[0]:
                    cv2.circle(
                        image_original_copy_2,
                        (centroid_x, centroid_y),
                        r,
                        (0, 255, 0),
                        2,
                    )

                    if visualization_flag:
                        print("Radius: ", r)
                        # show the result
                        cv2.imshow("Circles", image_original_copy_2)
                        cv2.waitKey(0)
                        cv2.destroyAllWindows()
            else:
                rospy.loginfo("[plug removal] red circle not found !")
                return None, None

        # get the biggest circle from the list
        if circles is not None:
            circles = np.uint16(np.around(circles))
            circles = sorted(circles[0], key=lambda x: x[2], reverse=True)
            (centroid_x, centroid_y, r) = circles[0]
            cv2.circle(
                image_original, (centroid_x, centroid_y), r, (0, 255, 0), 2
            )

            if visualization_flag:
                # show the result
                cv2.imshow("Final Circles", image_original)
                cv2.waitKey(0)
                cv2.destroyAllWindows()
        else:
            rospy.loginfo("[plug removal] red circle not found !")
            return None, None

        # display the center of the circle
        cv2.circle(image_original, (centroid_x, centroid_y), 5, (0, 0, 255), -1)

        # display the image with target points
        cv2.circle(image_original, (target_x, target_y), 5, (255, 255, 0), -1)

        # draw a horizontal line from the target point
        cv2.line(
            image_original,
            (0, target_y),
            (image.shape[1], target_y),
            (0, 0, 255),
            2,
        )
        # draw a vertical line from the target point
        cv2.line(
            image_original,
            (target_x, 0),
            (target_x, image.shape[0]),
            (0, 0, 255),
            2,
        )

        # calculate the error
        error_x = target_x - centroid_x
        error_y = target_y - centroid_y

        # print the error in the image
        cv2.putText(
            image_original,
            "Error: {}, {}".format(error_x, error_y),
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 0),
            2,
        )

        # draw a horizontal line from the centroid to the target point (x-axis only)
        horizontal_line = [(centroid_x, centroid_y), (target_x, centroid_y)]
        cv2.line(
            image_original,
            horizontal_line[0],
            horizontal_line[1],
            (0, 255, 0),
            2,
        )

        # draw a vertical line from the end of the horizontal line to the target point (y-axis only)
        vertical_line = [(target_x, centroid_y), (target_x, target_y)]
        cv2.line(
            image_original, vertical_line[0], vertical_line[1], (0, 255, 0), 2
        )

        if visualization_flag:
            # show the result
            cv2.imshow("FINAL IMAGE", image_original)
            cv2.waitKey(0)
            cv2.destroyAllWindows()

        self.img_pub.publish(self.bridge.cv2_to_imgmsg(image_original, "bgr8"))

        return error_x, error_y

    def align_red_port(self, save_debug_images=False):
        min_x = 200
        max_x = 1000
        min_y = 100
        max_y = 520
        ## These are targets when tool_pose_z = approx 0.148 m
        ## if robot ends up too far back, increase target_y
        ## if robot ends up too far right, increase target_x
        target_x = 559
        target_y = 318
        color_img = self.image[min_y:max_y, min_x:max_x]
        img = cv2.cvtColor(color_img, cv2.COLOR_BGR2GRAY)
        mask = cv2.GaussianBlur(img, (3, 3), sigmaX=33)
        mask = cv2.adaptiveThreshold(
            img, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 11, 23
        )
        mask = erode(mask)
        mask = dilate(mask)
        allc, vcirc, hcirc = self.detect_silver_circles(mask)
        if allc is None:
            return None, None
        for idx, cc in enumerate(allc):
            a, b, r = cc[0], cc[1], cc[2]
            cv2.circle(color_img, (a, b), r, (0, 0, 255), 2)
        avg_x = 0
        avg_y = 0
        if len(vcirc) == 1:
            avg_x = (allc[vcirc[0][0]][0] + allc[vcirc[0][1]][0]) / 2.0
            error_x = target_x - avg_x
        else:
            error_x = None
        if len(hcirc) == 1:
            avg_y = (allc[hcirc[0][0]][1] + allc[hcirc[0][1]][1]) / 2.0
            error_y = target_y - avg_y
        else:
            error_y = None
        if error_x is not None:
            rospy.loginfo(
                "Vcirc Centroid X: %d, %d, Error: %d" % (avg_x, avg_y, error_x)
            )
        if error_y is not None:
            rospy.loginfo(
                "Hcirc Centroid Y: %d, %d, Error: %d" % (avg_x, avg_y, error_y)
            )

        for pair in vcirc:
            cc1 = allc[pair[0]]
            cc2 = allc[pair[1]]
            cv2.circle(color_img, (cc1[0], cc1[1]), cc1[2], (255, 0, 0), 2)
            cv2.circle(color_img, (cc2[0], cc2[1]), cc2[2], (255, 0, 0), 2)

        for pair in hcirc:
            cc1 = allc[pair[0]]
            cc2 = allc[pair[1]]
            cv2.circle(color_img, (cc1[0], cc1[1]), cc1[2], (0, 255, 0), 2)
            cv2.circle(color_img, (cc2[0], cc2[1]), cc2[2], (0, 255, 0), 2)
        self.img_pub.publish(self.bridge.cv2_to_imgmsg(color_img, "bgr8"))
        return error_x, error_y

    def detect_silver_circles(self, img):
        imgblur = cv2.blur(img, (3, 3))
        detected_circles = cv2.HoughCircles(
            imgblur,
            cv2.HOUGH_GRADIENT,
            1,
            20,
            param1=50,
            param2=15,
            minRadius=10,
            maxRadius=20,
        )
        if detected_circles is None:
            return None, None, None
        detected_circles = detected_circles.astype(np.int64)[0]
        filtered_circles = []
        wanted_circle_centers = np.array(
            [[535, 268], [531, 164], [324, 342], [218, 343]]
        )
        circ_centers = detected_circles[:, :2]
        distances = spatial.distance.cdist(circ_centers, circ_centers)
        np.fill_diagonal(distances, 1000.0)
        vertical_circle_idx = []
        horizontal_circle_idx = []
        dist_threshold = 10
        axis_threshold = 5
        # approximate distance in pixels between silver circles (screws) at 0.148m height
        target_dist = 104
        all_circles = detected_circles.copy()
        selected_circles = []
        for idx, (cc, dist) in enumerate(zip(detected_circles, distances)):
            # find circle which is at target_dist from current circle
            best_match_idx = np.argmin(abs(dist - target_dist))
            bmc = detected_circles[best_match_idx]
            smallest = dist[np.argmin(abs(dist - target_dist))]
            # if its close enough to target_dist and this distance is along the x-coordinate (horizontal circles)
            if abs(smallest - target_dist) < dist_threshold and (
                abs(abs(cc[0] - bmc[0]) - target_dist) < axis_threshold
            ):
                filtered_circles.append(cc)
                if idx not in selected_circles:
                    horizontal_circle_idx.append([idx, best_match_idx])
                    selected_circles.append(best_match_idx)
            # if its close enough to target_dist and this distance is along the y-coordinate (horizontal circles)
            if abs(smallest - target_dist) < dist_threshold and (
                abs(abs(cc[1] - bmc[1]) - target_dist) < axis_threshold
            ):
                filtered_circles.append(cc)
                if idx not in selected_circles:
                    vertical_circle_idx.append([idx, best_match_idx])
                    selected_circles.append(best_match_idx)
        detected_circles = np.array(filtered_circles)
        # if we get two pairs of horizontal circles, pick the one that's above
        if len(horizontal_circle_idx) == 2:
            if (
                all_circles[horizontal_circle_idx[0][0]][1]
                < all_circles[horizontal_circle_idx[1][0]][1]
            ):
                horizontal_circle_idx.pop(1)
            else:
                horizontal_circle_idx.pop(0)
        return all_circles, vertical_circle_idx, horizontal_circle_idx

    def move_down_velocity_control(self):
        linear_vel_z = rospy.get_param("~linear_vel_z", 0.005)
        force_z_diff_threshold = rospy.get_param("~force_z_diff_threshold", 3.0)
        force_control_loop_rate = rospy.Rate(
            rospy.get_param("~force_control_loop_rate", 10.0)
        )
        stop = False
        self.current_force_z = []
        num_retries = 0
        while not rospy.is_shutdown():
            if len(self.current_force_z) < 20:
                num_retries += 1
                if num_retries > 100:
                    rospy.logerr("No force measurements received")
                    break
                force_control_loop_rate.sleep()
                continue
            msg = kortex_driver.msg.TwistCommand()
            msg.reference_frame = (
                kortex_driver.msg.CartesianReferenceFrame.CARTESIAN_REFERENCE_FRAME_TOOL
            )
            msg.twist.linear_z = linear_vel_z
            # print("Force: {}".format(abs(np.mean(self.current_force_z) - self.current_force_z[-1])))
            if (
                abs(np.mean(self.current_force_z) - self.current_force_z[-1])
                > force_z_diff_threshold
            ):
                rospy.loginfo("Force difference threshold reached")
                stop = True
                msg.twist.linear_z = 0.0
            self.cart_vel_pub.publish(msg)
            if stop:
                break
            force_control_loop_rate.sleep()
        msg = kortex_driver.msg.TwistCommand()
        msg.reference_frame = (
            kortex_driver.msg.CartesianReferenceFrame.CARTESIAN_REFERENCE_FRAME_TOOL
        )
        msg.twist.linear_z = -linear_vel_z
        for idx in range(5):
            self.cart_vel_pub.publish(msg)
            force_control_loop_rate.sleep()
        msg.twist.linear_z = 0.0
        self.cart_vel_pub.publish(msg)
        force_control_loop_rate.sleep()
        rospy.sleep(0.1)
        return True

    def move_down_insert(self, grasp_height):
        linear_vel_z = rospy.get_param("~linear_vel_z", 0.005)
        force_z_diff_threshold = 10.0
        force_control_loop_rate = rospy.Rate(
            rospy.get_param("~force_control_loop_rate", 10.0)
        )
        plug_insertion_height_threshold = rospy.get_param(
            "~plug_insertion_height_threshold", 0.124
        )
        stop = False
        self.current_force_z = []
        num_retries = 0
        while not rospy.is_shutdown():
            if len(self.current_force_z) < 20:
                num_retries += 1
                if num_retries > 100:
                    rospy.logerr("No force measurements received")
                    break
                force_control_loop_rate.sleep()
                continue
            msg = kortex_driver.msg.TwistCommand()
            msg.reference_frame = (
                kortex_driver.msg.CartesianReferenceFrame.CARTESIAN_REFERENCE_FRAME_TOOL
            )
            msg.twist.linear_z = linear_vel_z
            # print("Force: {}".format(abs(np.mean(self.current_force_z) - self.current_force_z[-1])))
            if (
                abs(np.mean(self.current_force_z) - self.current_force_z[-1])
                > force_z_diff_threshold
            ):
                rospy.loginfo("Force difference threshold reached")
                stop = True
                msg.twist.linear_z = 0.0
            if self.current_height < grasp_height + 0.002:
                rospy.loginfo("Height difference threshold reached")
                stop = True
                msg.twist.linear_z = 0.0
            self.cart_vel_pub.publish(msg)
            if stop:
                break
            force_control_loop_rate.sleep()
        inserted_plug = False
        if (
            self.current_height < plug_insertion_height_threshold
        ):  # we've definitely inserted the plug
            rospy.loginfo("Height threshold reached; we have inserted the plug")
            inserted_plug = True

            rotate_plug = True
            if rotate_plug:
                current_pose = self.arm.get_current_pose()
                current_pose.theta_z_deg += 37
                self.arm.send_cartesian_pose(current_pose)
                rospy.loginfo("Rotated plug")
            self.arm.execute_gripper_command(0.3)  # open gripper

        msg = kortex_driver.msg.TwistCommand()
        msg.reference_frame = (
            kortex_driver.msg.CartesianReferenceFrame.CARTESIAN_REFERENCE_FRAME_MIXED
        )
        self.cart_vel_pub.publish(msg)
        force_control_loop_rate.sleep()

        current_pose = self.arm.get_current_pose()
        current_pose.z = 0.1475
        self.arm.send_cartesian_pose(current_pose)
        if not inserted_plug:
            # if we fail move backwards a bit to retry
            msg = kortex_driver.msg.TwistCommand()
            msg.reference_frame = (
                kortex_driver.msg.CartesianReferenceFrame.CARTESIAN_REFERENCE_FRAME_TOOL
            )
            msg.twist.linear_y = -0.005
            msg.twist.linear_x = -0.005
            for idx in range(10):
                self.cart_vel_pub.publish(msg)
                force_control_loop_rate.sleep()
        msg = kortex_driver.msg.TwistCommand()
        msg.reference_frame = (
            kortex_driver.msg.CartesianReferenceFrame.CARTESIAN_REFERENCE_FRAME_MIXED
        )
        self.cart_vel_pub.publish(msg)
        force_control_loop_rate.sleep()

        return inserted_plug

    def move_arm_2D_space(self, direction):
        msg = kortex_driver.msg.TwistCommand()
        msg.reference_frame = (
            kortex_driver.msg.CartesianReferenceFrame.CARTESIAN_REFERENCE_FRAME_TOOL
        )
        if direction == 0:
            rospy.loginfo("Stopped moving")
            msg.twist.linear_x = 0.0
            self.cart_vel_pub.publish(msg)
            self.loop_rate.sleep()
            self.stop = True
            self.cart_vel_pub.publish(msg)
        elif direction == -1 or direction == 1:
            rospy.loginfo("Moving in y direction")
            msg.twist.linear_x = direction * self.velocity
            self.cart_vel_pub.publish(msg)
            self.loop_rate.sleep()
        else:
            rospy.loginfo("Invalid direction")

    def move_up_velocity_control(self):
        rospy.loginfo("Moving up")
        pre_height_above_button = rospy.get_param(
            "~pre_height_above_button", 0.20
        )
        msg = kortex_driver.msg.TwistCommand()
        msg.reference_frame = (
            kortex_driver.msg.CartesianReferenceFrame.CARTESIAN_REFERENCE_FRAME_TOOL
        )
        for idx in range(50):
            if self.current_height > 0.1475:
                break
            msg.twist.linear_z = -0.01
            self.cart_vel_pub.publish(msg)
            self.loop_rate.sleep()
        msg.twist.linear_z = 0.0
        self.cart_vel_pub.publish(msg)
        self.loop_rate.sleep()
        self.cart_vel_pub.publish(msg)

    def move_forward(self):
        rospy.loginfo("Moving forward")
        msg = kortex_driver.msg.TwistCommand()
        msg.reference_frame = (
            kortex_driver.msg.CartesianReferenceFrame.CARTESIAN_REFERENCE_FRAME_TOOL
        )
        for idx in range(23):
            msg.twist.linear_y = 0.01
            self.cart_vel_pub.publish(msg)
            self.loop_rate.sleep()
        msg.twist.linear_y = 0.0
        self.cart_vel_pub.publish(msg)
        self.loop_rate.sleep()
        self.cart_vel_pub.publish(msg)

    def save_debug_images(self):
        rospy.loginfo_once("Saving debug images")

        # check if the directory exists
        if not os.path.exists(self.save_debug_images_dir):
            pass
            # TODO: create a directory
        else:
            # get the current date and time
            now = datetime.datetime.now()
            date_time = now.strftime("%m_%d_%Y_%H_%M_%S")

            # save the images
            cv2.imwrite(
                os.path.join(
                    self.save_debug_images_dir, date_time + "_rgb.png"
                ),
                self.image,
            )
