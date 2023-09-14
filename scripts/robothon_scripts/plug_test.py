#!/usr/bin/env python3
from __future__ import print_function

import tf
import sys
import rospy
import copy
from std_msgs.msg import String
from kinova_apps.full_arm_movement import FullArmMovement
from kinova_apps.plug_remove_slid_action import PlugRemoveSlidAction
from kinova_apps.transform_utils import TransformUtils
from utils.kinova_pose import get_kinovapose_from_list
import numpy as np
import actionlib


class RobothonTask(object):

    """pick and place code using full arm movement and 3d segmentation perception"""

    def __init__(self):
        self.joint_angles = rospy.get_param("~joint_angles", None)
        self.perceive_board_pose = rospy.get_param("~perceive_board_pose")
        self.arm = FullArmMovement()
        self.tu = TransformUtils()
        self.remove_slid_action = PlugRemoveSlidAction(self.arm, self.tu)

        self.loop_rate = rospy.Rate(10.0)

        self.setup_arm_for_pick()
        # rospy.sleep(3.0)
        rospy.loginfo("READY!")

    def setup_arm_for_pick(self):
        """Setup the arm to go to pick pose
        :returns: None

        """
        self.arm.clear_faults()
        self.arm.subscribe_to_a_robot_notification()
        # self.arm.test_send_joint_angles(self.joint_angles["vertical_pose"])
        # self.arm.send_joint_angles(self.joint_angles["perceive_table"])
        perceive_board_pose = get_kinovapose_from_list(self.perceive_board_pose)
        self.arm.send_cartesian_pose(perceive_board_pose)
        rospy.sleep(0.1)
        self.arm.execute_gripper_command(0.35)  # Open the gripper
        # self.arm.example_send_gripper_command(0.5) #half close the gripper
        # self.arm.execute_gripper_command(0.9) #full close the gripper
        rospy.sleep(0.1)

    def test(self):
        self.remove_slid_action.do()
        rospy.sleep(0.1)
        pass


if __name__ == "__main__":
    rospy.init_node("robothon_task")
    task = RobothonTask()
    task.test()
    # PAP.test_go_to_board()
    # PAP.test_press_button()
    rospy.spin()
