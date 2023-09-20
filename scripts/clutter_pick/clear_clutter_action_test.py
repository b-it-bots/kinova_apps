#!/usr/bin/env python3
from __future__ import print_function

import rospy

from kortex_driver.srv import *
from kortex_driver.msg import *

from kinova_apps.full_arm_movement import FullArmMovement
from utils.transform_utils import TransformUtils

from kinova_apps.clutter_pick.clear_clutter_action import ClearClutterAction


class ClearClutterTest(object):

    """ClearClutter test using full arm movement"""

    def __init__(self):
        self.fam = FullArmMovement()
        self.transform_utils = TransformUtils()
        self.cc_action = ClearClutterAction(self.fam, self.transform_utils)
        self.joint_angles = rospy.get_param("joint_angles", None)
        self.setup_arm_for_pick()

    def setup_arm_for_pick(self):
        """Setup the arm to go to pick pose
        :returns: None

        """
        self.fam.clear_faults()
        self.fam.subscribe_to_a_robot_notification()
        # self.fam.send_joint_angles(self.joint_angles["perceive_table"])
        self.fam.execute_gripper_command(0.0)

    def do(self):
        if self.cc_action.do():
            return True


if __name__ == "__main__":
    rospy.init_node("probe_test")
    cc_test = ClearClutterTest()
    cc_test.do()
    rospy.spin()
