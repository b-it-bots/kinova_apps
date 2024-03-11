import sys
import rospy
import time
from kortex_driver.srv import *
from kortex_driver.msg import *

from kinova_apps.abstract_action import AbstractAction

from kinova_apps.full_arm_movement import FullArmMovement
from utils.transform_utils import TransformUtils

from utils.kinova_pose import KinovaPose

from copy import deepcopy

class DMPTrajectoryAction(AbstractAction):
    def __init__(self, arm: FullArmMovement, transform_utils: TransformUtils):
        super().__init__(arm, transform_utils)

        rospy.loginfo("DMP Trajectory Action")

    def pre_perceive(self) -> bool:
        return True

    def act(self) -> bool:
        success = True

        # Prepare and send pose 1
        current_pose = self.arm.get_current_pose()

        pose = deepcopy(current_pose)

        pose.z += 0.1

        success &= self.arm.send_constrained_pose(pose)

        # Prepare and send pose 2
        pose.z -= 0.1

        success &= self.arm.send_constrained_pose(pose)


        return success

    def verify(self) -> bool:
        return True
