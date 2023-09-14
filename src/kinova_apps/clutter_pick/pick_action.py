#!/usr/bin/env python3

import rospy

import kortex_driver.msg

from kinova_apps.abstract_action import AbstractAction
from kinova_apps.full_arm_movement import FullArmMovement
from utils.transform_utils import TransformUtils


class PickAction(AbstractAction):
    def __init__(
        self,
        arm: FullArmMovement,
        transform_utils: TransformUtils,
        reference_frame: str = "base_link",
    ) -> None:
        super().__init__(arm, transform_utils)

        self.reference_frame = reference_frame

    def pre_perceive(self) -> bool:
        success = True
        # open gripper before picking
        success &= self.arm.execute_gripper_command(1.0)

        return success

    def act(self) -> bool:
        success = True
        # get the current pose of the end effector
        current_pose = self.arm.get_current_pose()

        # create a new pose with 10cm away in x direction
        new_pose = current_pose
        new_pose.x += 0.1

        # move to the new pose
        success &= self.arm.send_cartesian_pose(new_pose)

        success &= self.arm.execute_gripper_command(0.0)

        return success

    def verify(self) -> bool:
        print("in verify")
        return True
    