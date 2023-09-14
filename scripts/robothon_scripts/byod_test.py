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
from kinova_apps.byod_action import ByodAction
from kinova_apps.transform_utils import TransformUtils
import kortex_driver.msg
import numpy as np
from kortex_driver.srv import *
from kortex_driver.msg import *
import actionlib

from utils.force_measure import ForceMeasurmement


class ByodTest(object):
    def __init__(self):
        print("full arm movement")
        self.fam = FullArmMovement()
        self.transform_utils = TransformUtils()
        self.fm = ForceMeasurmement()
        self.setup_arm()
        self.byod_action = ByodAction(self.fam, self.transform_utils)

    def setup_arm(self):
        """Setup the arm
        :returns: None

        """
        # clear faults and subscribe to robot notifications to get feedback from the robot
        self.fam.clear_faults()
        self.fam.subscribe_to_a_robot_notification()

    def test(self):
        self.byod_action.do()


if __name__ == "__main__":
    rospy.init_node("BYOD Test")
    byod_test = ByodTest()
    byod_test.test()
