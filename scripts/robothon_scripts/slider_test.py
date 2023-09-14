#!/usr/bin/env python3
from __future__ import print_function

import tf
import sys
import rospy
import copy
import math
from geometry_msgs.msg import PoseStamped, Quaternion
from std_msgs.msg import String
from kinova_apps.full_arm_movement import FullArmMovement
from kinova_apps.slider_action import SliderAction
from kinova_apps.transform_utils import TransformUtils
import kortex_driver.msg
import numpy as np
from kortex_driver.srv import *
from kortex_driver.msg import *
import actionlib

from utils.force_measure import ForceMeasurmement


class SliderTest(object):
    def __init__(self):
        print("full arm movement")
        self.fam = FullArmMovement()
        self.transform_utils = TransformUtils()
        self.joint_angles = rospy.get_param("~joint_angles", None)
        self.fm = ForceMeasurmement()
        self.setup_arm()
        self.slider_action = SliderAction(self.fam, self.transform_utils)

    def setup_arm(self):
        """Setup the arm
        :returns: None

        """
        # clear faults and subscribe to robot notifications to get feedback from the robot
        self.fam.clear_faults()
        self.fam.subscribe_to_a_robot_notification()
        self.fam.send_joint_angles(self.joint_angles["perceive_table"])

    def test(self):
        self.slider_action.do()


if __name__ == "__main__":
    rospy.init_node("Slider Test")
    slider_test = SliderTest()
    slider_test.test()
