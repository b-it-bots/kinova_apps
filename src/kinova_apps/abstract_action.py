#!/usr/bin/env python3
#
import rospy
from abc import ABC, abstractmethod
from typing import Type, TypeVar
from kinova_apps.full_arm_movement import FullArmMovement
from utils.transform_utils import TransformUtils


class AbstractAction(ABC):
    # ToDo use TypeVar and add typing for arm variable
    def __init__(
        self, arm: FullArmMovement, transform_utils: TransformUtils
    ) -> None:
        self.arm = arm
        self.transform_utils = transform_utils

    @abstractmethod
    def pre_perceive(self) -> bool:
        pass

    @abstractmethod
    def act(self) -> bool:
        pass

    @abstractmethod
    def verify(self) -> bool:
        pass

    def do(self) -> bool:
        success = True

        success &= self.pre_perceive()
        success &= self.act()
        success &= self.verify()

        if not success:
            rospy.logerr("The action encountered an error")

        return success
