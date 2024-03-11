#!/usr/bin/env python3
from __future__ import print_function

import rospy

from kortex_driver.srv import *
from kortex_driver.msg import *

from kinova_apps.full_arm_movement import FullArmMovement
from utils.transform_utils import TransformUtils

from kinova_apps.dmp.traj_action import DMPTrajectoryAction


class DMPTrajActionTesst(object):

    def __init__(self):
        self.fam = FullArmMovement()
        self.transform_utils = TransformUtils()
        self.dmp_traj_action = DMPTrajectoryAction(self.fam, self.transform_utils)
        self.setup_arm()

    def setup_arm(self):
        self.fam.clear_faults()
        self.fam.set_cartesian_reference_frame()
        self.fam.subscribe_to_a_robot_notification()

    def do(self):
        if self.dmp_traj_action.do():
            return True
        
if __name__ == "__main__":
    rospy.init_node("dmp_traj_action_test")
    dmp_traj_action_test = DMPTrajActionTesst()
    dmp_traj_action_test.do()