#!/usr/bin/env python3

import rospy

from sensor_msgs.msg import Image

import kortex_driver.msg

from kinova_apps.abstract_action import AbstractAction
from kinova_apps.full_arm_movement import FullArmMovement
from utils.transform_utils import TransformUtils


class ClearClutterAction(AbstractAction):
    def __init__(
        self,
        arm: FullArmMovement,
        transform_utils: TransformUtils,
        reference_frame: str = "base_link",
    ) -> None:
        super().__init__(arm, transform_utils)

        self.reference_frame = reference_frame

        self.cartesian_velocity_pub = rospy.Publisher(
            "/my_gen3/in/cartesian_velocity", kortex_driver.msg.TwistCommand, queue_size=1
        )

        self.rgb_topic = 'input_image_topic'
        self.depth_topic = 'input_pc_topic'

    def pre_perceive(self) -> bool:
        success = True
        # open gripper before picking
        success &= self.arm.execute_gripper_command(1.0)

        return success

    def act(self) -> bool:
        success = True
        
        # subscribe to rgb 
        self.rgb_image = rospy.wait_for_message(self.rgb_topic, Image, timeout=10)

        # print image properties
        print("image width: ", self.rgb_image.width)

        return success

    def verify(self) -> bool:
        print("in verify")
        return True
    
    def rgb_callback(self, msg):
        print("in rgb_callback")
        # unsubscribe from rgb
        self.rgb_sub.unregister()