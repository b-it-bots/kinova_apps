#!/usr/bin/env python3

import rospy

import sys
import numpy as np
import cv2
import cv_bridge
import matplotlib.pyplot as plt

from sensor_msgs.msg import Image


class CalibrateHSV:
    def __init__(self) -> None:
        self.bridge = cv_bridge.CvBridge()
        self.image_sub = rospy.Subscriber(
            "/camera/color/image_raw", Image, self.im_sub
        )
        self.img = None

    def nothing(self, x):
        pass

    def im_sub(self, msg):
        # convert image to cv2
        self.img = self.bridge.imgmsg_to_cv2(
            msg, desired_encoding="passthrough"
        )

        # convert to bgr
        self.img = cv2.cvtColor(self.img, cv2.COLOR_RGB2BGR)

    def calibrate_hsv(self, image=None):
        # code from https://stackoverflow.com/a/57263140

        # Create a window
        cv2.namedWindow("image")

        # create trackbars for color change
        cv2.createTrackbar(
            "HMin", "image", 0, 179, self.nothing
        )  # Hue is from 0-179 for Opencv
        cv2.createTrackbar("SMin", "image", 0, 255, self.nothing)
        cv2.createTrackbar("VMin", "image", 0, 255, self.nothing)
        cv2.createTrackbar("HMax", "image", 0, 179, self.nothing)
        cv2.createTrackbar("SMax", "image", 0, 255, self.nothing)
        cv2.createTrackbar("VMax", "image", 0, 255, self.nothing)

        # Set default value for MAX HSV trackbars.
        cv2.setTrackbarPos("HMax", "image", 179)
        cv2.setTrackbarPos("SMax", "image", 255)
        cv2.setTrackbarPos("VMax", "image", 255)

        # Initialize to check if HSV min/max value changes
        hMin = sMin = vMin = hMax = sMax = vMax = 0
        phMin = psMin = pvMin = phMax = psMax = pvMax = 0

        img = image
        output = img
        waitTime = 33

        while 1 and self.img is not None and not rospy.is_shutdown():
            if image is None:
                img = self.img

            # get current positions of all trackbars
            hMin = cv2.getTrackbarPos("HMin", "image")
            sMin = cv2.getTrackbarPos("SMin", "image")
            vMin = cv2.getTrackbarPos("VMin", "image")

            hMax = cv2.getTrackbarPos("HMax", "image")
            sMax = cv2.getTrackbarPos("SMax", "image")
            vMax = cv2.getTrackbarPos("VMax", "image")

            # Set minimum and max HSV values to display
            lower = np.array([hMin, sMin, vMin])
            upper = np.array([hMax, sMax, vMax])

            # Create HSV Image and threshold into a range.
            hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
            mask = cv2.inRange(hsv, lower, upper)
            output = cv2.bitwise_and(img, img, mask=mask)

            # Print if there is a change in HSV value
            if (
                (phMin != hMin)
                | (psMin != sMin)
                | (pvMin != vMin)
                | (phMax != hMax)
                | (psMax != sMax)
                | (pvMax != vMax)
            ):
                print(
                    "(hMin = %d , sMin = %d, vMin = %d), (hMax = %d , sMax = %d, vMax = %d)"
                    % (hMin, sMin, vMin, hMax, sMax, vMax)
                )
                phMin = hMin
                psMin = sMin
                pvMin = vMin
                phMax = hMax
                psMax = sMax
                pvMax = vMax

            # Display output image
            cv2.imshow("image", output)

            # Wait longer to prevent freeze for videos.
            if cv2.waitKey(waitTime) & 0xFF == ord("q"):
                break

            # if window is closed, exit program
            if cv2.getWindowProperty("image", 1) == -1:
                break

        cv2.destroyAllWindows()
        # stop node
        rospy.signal_shutdown("Calibration complete")


if __name__ == "__main__":
    # process_image_with_cubes()
    rospy.init_node("calibrate_hsv")
    calibrate_hsv = CalibrateHSV()
    calibrate_hsv.calibrate_hsv()
    rospy.spin()
