#!/usr/bin/env python3
#
import rospy
import tf
from kinova_apps.full_arm_movement import FullArmMovement
from kinova_apps.abstract_action import AbstractAction
from geometry_msgs.msg import PoseStamped, Quaternion
import kortex_driver.msg
import math
from utils.transform_utils import TransformUtils
from utils.kinova_pose import get_kinovapose_from_pose_stamped
import numpy as np


class ButtonPressAction(AbstractAction):
    """
    Assumption: From top view the approximate pose of the
    buttons is available
P
    - Move the arm to the top of the buttons
    - Do visual servoving and adjust arm
    - Move arm with velocity control
    - stop when the force is higher
    - retreat above
    """

    def __init__(
        self,
        arm: FullArmMovement,
        transform_utils: TransformUtils,
        reference_frame: str = "board_link",
    ) -> None:
        super(ButtonPressAction, self).__init__(arm, transform_utils)
        self.base_feedback_sub = rospy.Subscriber(
            "/my_gen3/base_feedback",
            kortex_driver.msg.BaseCyclic_Feedback,
            self.base_feedback_cb,
        )
        self.cart_vel_pub = rospy.Publisher(
            "/my_gen3/in/cartesian_velocity",
            kortex_driver.msg.TwistCommand,
            queue_size=1,
        )
        self.current_force_z = []
        self.button_reference_frame = reference_frame

    def base_feedback_cb(self, msg):
        self.current_force_z.append(msg.base.tool_external_wrench_force_z)
        if len(self.current_force_z) > 25:
            self.current_force_z.pop(0)

    def pre_perceive(self) -> bool:
        rospy.loginfo("[%s] pre-preceive" % self.__class__.__name__)

        pre_height_above_button = rospy.get_param(
            "~pre_height_above_button", 0.02
        )
        if "red_button" in self.button_reference_frame:
            x_offset = 0.01
            pre_height_above_button += (
                0.07  # moving to safe height to avoid hitting the door
            )
        else:
            pre_height_above_button += 0.02
            x_offset = 0.0
        kinova_pose = self.transform_utils.transform_pose_frame_name(
            reference_frame_name=self.button_reference_frame,
            target_frame_name="base_link",
            offset_linear=[x_offset, 0.005, pre_height_above_button],
            offset_rotation_euler=[math.pi, 0.0, math.pi / 4],
        )

        self.arm.send_cartesian_pose(kinova_pose)
        self.arm.execute_gripper_command(1.0)
        if "red_button" in self.button_reference_frame:
            kinova_pose.z -= 0.09
            self.arm.send_cartesian_pose(kinova_pose)
        else:
            kinova_pose.z -= 0.04
            self.arm.send_cartesian_pose(kinova_pose)
        return True

    def act(self) -> bool:
        rospy.loginfo("[%s] starting action" % self.__class__.__name__)
        linear_vel_z = rospy.get_param("~linear_vel_z", 0.005)
        force_z_diff_threshold = rospy.get_param("~force_z_diff_threshold", 3.0)
        force_control_loop_rate = rospy.Rate(
            rospy.get_param("~force_control_loop_rate", 10.0)
        )
        stop = False
        self.current_force_z = []
        num_retries = 0
        while not rospy.is_shutdown():
            if len(self.current_force_z) < 20:
                num_retries += 1
                if num_retries > 100:
                    rospy.logerr("No force measurements received")
                    break
                force_control_loop_rate.sleep()
                continue
            msg = kortex_driver.msg.TwistCommand()
            msg.reference_frame = (
                kortex_driver.msg.CartesianReferenceFrame.CARTESIAN_REFERENCE_FRAME_TOOL
            )
            msg.twist.linear_z = linear_vel_z
            if (
                abs(np.mean(self.current_force_z) - self.current_force_z[-1])
                > force_z_diff_threshold
            ):
                stop = True
                msg.twist.linear_z = 0.0
                msg.reference_frame = (
                    kortex_driver.msg.CartesianReferenceFrame.CARTESIAN_REFERENCE_FRAME_MIXED
                )  # publish 0 vel in mixed frame
            self.cart_vel_pub.publish(msg)
            if stop:
                break
            force_control_loop_rate.sleep()
        board_height = self.arm.get_current_pose().z
        rospy.set_param("/board_height", board_height)
        msg = kortex_driver.msg.TwistCommand()
        msg.reference_frame = (
            kortex_driver.msg.CartesianReferenceFrame.CARTESIAN_REFERENCE_FRAME_MIXED
        )  # publish 0 vel in mixed frame
        msg.twist.linear_z = 0.0
        self.cart_vel_pub.publish(msg)
        force_control_loop_rate.sleep()

        current_pose = self.arm.get_current_pose()
        current_pose.z += 0.05
        if "red_button" in self.button_reference_frame:
            current_pose.z += 0.1
        self.arm.send_cartesian_pose(current_pose)
        return True

    def verify(self) -> bool:
        print("in verify")
        pre_height_above_button = rospy.get_param(
            "~pre_height_above_button", 0.05
        )
        kinova_pose = self.transform_utils.transform_pose_frame_name(
            reference_frame_name="gui_link",
            target_frame_name="base_link",
            offset_linear=[0.0, 0.0, pre_height_above_button],
            offset_rotation_euler=[math.pi, 0.0, math.pi / 2],
        )

        # self.arm.send_cartesian_pose(kinova_pose)
        return True
