#!/usr/bin/env python3


"""
Author: Kevin Patel
Date: 14-April-2023

Task: Visual Servoing for Kinova arm using OpenCV

Steps:
1. Detect the desired object using OpenCV
2. Calculate the error between the current position of the object and the desired position (center of the image)

Assumptions:
1. The desired object is a black color
2. The desired object is a kind of circular shape

TODO:
1. Take 30 images and calculate the median of the error
2. Pose transformation for moving direction

"""

import rospy
import numpy as np
import kortex_driver.msg
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError
import cv2
from kinova_apps.full_arm_movement import FullArmMovement


class WrenchTest(object):
    def __init__(self):
        self.arm = FullArmMovement()
        self.sub = rospy.Subscriber(
            "/my_gen3/base_feedback",
            kortex_driver.msg.BaseCyclic_Feedback,
            self.base_feedback_cb,
        )
        self.pub = rospy.Publisher(
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
        self.loop_rate = rospy.Rate(3.0)
        self.current_force_z = []
        self.bridge = CvBridge()
        self.error = 0.0
        self.error_threshold = 5.0
        self.stop = False
        self.velocity = 0.005
        self.image_queue = []
        self.move_up_done = False
        self.move_down_done = False
        self.close_gripper_done = False

    def base_feedback_cb(self, msg):
        self.current_force_z.append(msg.base.tool_external_wrench_force_z)
        if len(self.current_force_z) > 25:
            self.current_force_z.pop(0)

    def run(self):
        stop = False
        while not rospy.is_shutdown():
            if len(self.current_force_z) < 20:
                self.loop_rate.sleep()
                continue
            msg = kortex_driver.msg.TwistCommand()
            msg.twist.linear_z = -0.01
            if (
                abs(np.mean(self.current_force_z) - self.current_force_z[-1])
                > 3.0
            ):
                stop = True
                msg.twist.linear_z = 0.0
            self.pub.publish(msg)
            if stop:
                break
            self.loop_rate.sleep()
        msg = kortex_driver.msg.TwistCommand()
        msg.twist.linear_z = 0.01
        for idx in range(5):
            self.pub.publish(msg)
            self.loop_rate.sleep()
        msg.twist.linear_z = 0.0
        self.pub.publish(msg)
        self.loop_rate.sleep()

    def move(self, direction):
        msg = kortex_driver.msg.TwistCommand()
        if direction == 0:
            rospy.loginfo("Stopped moving")
            msg.twist.linear_x = 0.0
            self.pub.publish(msg)
            self.loop_rate.sleep()
            self.stop = True
            self.pub.publish(msg)
        elif direction == -1 or direction == 1:
            rospy.loginfo("Moving in y direction")
            msg.twist.linear_x = direction * self.velocity
            self.pub.publish(msg)
            self.loop_rate.sleep()
        else:
            rospy.loginfo("Invalid direction")

    def image_cb(self, msg):
        # get the image from the message
        try:
            image = self.bridge.imgmsg_to_cv2(msg, "bgr8")
        except CvBridgeError as e:
            print(e)

        # store only the last 20 images
        if len(self.image_queue) > 20:
            self.image_queue.pop(0)
        self.image_queue.append(image)

    def run_visual_servoing(self):
        while not rospy.is_shutdown():
            if len(self.image_queue) == 0:
                self.loop_rate.sleep()
                continue

            image = self.image_queue[-1]

            circularity_threshold = 0.8
            contours_area_threshold = 100
            black_color_threshold = 60

            # find the contours
            # convert the image to grayscale
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            # apply gaussian blur to the image
            blur = cv2.GaussianBlur(gray, (5, 5), 0)
            # apply canny edge detection
            canny = cv2.Canny(blur, 50, 150)
            # find the contours
            contours, _ = cv2.findContours(
                canny, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
            )

            # draw a horizontal line in the middle of the image
            horizontal_line = [
                (0, image.shape[0] // 2),
                (image.shape[1], image.shape[0] // 2),
            ]
            cv2.line(
                image, horizontal_line[0], horizontal_line[1], (0, 0, 255), 2
            )

            # draw a vertical line in the middle of the image
            vertical_line = [
                (image.shape[1] // 2, 0),
                (image.shape[1] // 2, image.shape[0]),
            ]
            cv2.line(image, vertical_line[0], vertical_line[1], (0, 0, 255), 2)

            # filter out black contours
            filtered_contours = []
            mean_colors = []
            circularities = []
            for contour in contours:
                # Calculate area and perimeter of the contour
                area = cv2.contourArea(contour)
                perimeter = cv2.arcLength(contour, True)

                # filter out small contours
                if area < contours_area_threshold:
                    continue

                # Calculate circularity of the contour (reference: https://en.wikipedia.org/wiki/Roundness)
                circularity = (4 * np.pi * area) / (perimeter**2)

                # create a mask of the contour
                n_mask = np.zeros(image.shape[:2], dtype="uint8")
                cv2.drawContours(n_mask, [contour], 0, (255, 255, 255), -1)

                # get the mean color of the contour using the mask
                mean_color = cv2.mean(image, mask=n_mask)

                # filter out non black contours
                # threshold the circularity value to classify the contour as circular or not
                if (
                    mean_color[0] < black_color_threshold
                    and circularity > circularity_threshold
                ):
                    filtered_contours.append(contour)
                    mean_colors.append(mean_color[0])
                    circularities.append(circularity)

            # print("Number of filtered contours: {}".format(len(filtered_contours)))

            # NOTE: it should only be one contour
            # draw the filtered contour one by one on the image
            for i, contour in enumerate(filtered_contours):
                cv2.drawContours(image, [contour], -1, (255, 0, 0), 3)

                # calculate the centroid of the contour
                M = cv2.moments(contour)
                centroid_x = int(M["m10"] / M["m00"])
                centroid_y = int(M["m01"] / M["m00"])
                centroid = (centroid_x, centroid_y)
                cv2.circle(image, centroid, 5, (0, 0, 255), -1)

                # calculate the perpendicular distance between the vertical line and the centroid in the image (y-axis only)
                self.error = (image.shape[1] // 2) - centroid_x
                print("Error: {}".format(self.error))

                # TODO: put a max threshold on error to prevent the arm from moving too much
                # call the arm motion function
                if not self.stop:
                    if abs(self.error) > 2 and abs(self.error) < 300:
                        if self.error > 0:
                            self.move(1)
                            rospy.loginfo("Moving arm to the right")
                        else:
                            self.move(-1)
                            rospy.loginfo("Moving arm to the left")
                    else:
                        self.move(0)
                        continue
                else:
                    self.move(0)
                    return True

                # draw the error line on the image from the centroid to the vertical line
                error_line = [centroid, (centroid_x + self.error, centroid_y)]
                cv2.line(image, error_line[0], error_line[1], (0, 255, 0), 2)

                # publish the image
                self.img_pub.publish(self.bridge.cv2_to_imgmsg(image, "bgr8"))
                # rospy.loginfo("Published a final image!")

    def move_down(self):
        rospy.loginfo("Moving down")
        pre_height_above_button = rospy.get_param(
            "~pre_height_above_button", 1.00
        )
        msg = kortex_driver.msg.TwistCommand()
        msg.twist.linear_z = pre_height_above_button - 0.02
        self.pub.publish(msg)
        self.loop_rate.sleep()
        self.move_up_done = True
        return True

    def close_gripper(self):
        rospy.loginfo("Closing gripper")
        # self.arm.execute_gripper_command(0.35) #Open the gripper
        # self.arm.example_send_gripper_command(0.5) #half close the gripper
        self.arm.execute_gripper_command(0.5)  # full close the gripper
        rospy.sleep(0.1)
        return True

    def move_up(self):
        rospy.loginfo("Moving up")
        pre_height_above_button = rospy.get_param(
            "~pre_height_above_button", 1.00
        )
        msg = kortex_driver.msg.TwistCommand()
        msg.twist.linear_z = -pre_height_above_button
        self.pub.publish(msg)
        self.loop_rate.sleep()
        return True


def main():
    rospy.init_node("visual_servoing_kinova")
    wt = WrenchTest()
    servoing_flag = wt.run_visual_servoing()
    if servoing_flag:
        if wt.move_down():
            rospy.sleep(2.0)
            if wt.close_gripper():
                rospy.sleep(1.0)
                if wt.move_up():
                    rospy.sleep(1.0)
                    print("Visual Servoing is done!")
                    # shutdown the node
                    rospy.signal_shutdown("Visual Servoing is done!")

    try:
        rospy.spin()
    except KeyboardInterrupt:
        print("Shutting down")


if __name__ == "__main__":
    main()
