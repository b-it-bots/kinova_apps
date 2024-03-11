import sys
import rospy
import time
from kortex_driver.srv import *
from kortex_driver.msg import *

from kinova_apps.abstract_action import AbstractAction

from kinova_apps.full_arm_movement import FullArmMovement
from utils.transform_utils import TransformUtils


class DMPTrajectoryAction(AbstractAction):
    def __init__(self, arm: FullArmMovement, transform_utils: TransformUtils):
        super().__init__(arm, transform_utils)

        rospy.loginfo("DMP Trajectory Action")

    def pre_perceive(self) -> bool:
        return True

    def act(self) -> bool:
        success = True

        # Prepare and send pose 1
        my_cartesian_speed = CartesianSpeed()
        my_cartesian_speed.translation = 0.1  # m/s
        my_cartesian_speed.orientation = 15  # deg/s

        my_constrained_pose = ConstrainedPose()
        my_constrained_pose.constraint.oneof_type.speed.append(my_cartesian_speed)

        feedback = rospy.wait_for_message(
            "/" + self.arm.robot_name + "/base_feedback", BaseCyclic_Feedback
        )

        my_constrained_pose.target_pose.x = feedback.base.commanded_tool_pose_x
        my_constrained_pose.target_pose.y = feedback.base.commanded_tool_pose_y
        my_constrained_pose.target_pose.z = feedback.base.commanded_tool_pose_z + 0.1
        my_constrained_pose.target_pose.theta_x = (
            feedback.base.commanded_tool_pose_theta_x
        )
        my_constrained_pose.target_pose.theta_y = (
            feedback.base.commanded_tool_pose_theta_y
        )
        my_constrained_pose.target_pose.theta_z = (
            feedback.base.commanded_tool_pose_theta_z
        )

        req = ExecuteActionRequest()
        req.input.oneof_action_parameters.reach_pose.append(my_constrained_pose)
        req.input.name = "pose1"
        req.input.handle.action_type = ActionType.REACH_POSE
        req.input.handle.identifier = 1001

        rospy.loginfo("Sending pose 1...")
        self.last_action_notif_type = None
        try:
            self.arm.execute_action(req)
        except rospy.ServiceException:
            rospy.logerr("Failed to send pose 1")
            success = False
        else:
            rospy.loginfo("Waiting for pose 1 to finish...")

        self.arm.wait_for_action_end_or_abort()

        # Prepare and send pose 2
        req.input.handle.identifier = 1002
        req.input.name = "pose2"

        my_constrained_pose.target_pose.z = 0.3

        req.input.oneof_action_parameters.reach_pose[0] = my_constrained_pose

        rospy.loginfo("Sending pose 2...")
        self.last_action_notif_type = None
        try:
            self.arm.execute_action(req)
        except rospy.ServiceException:
            rospy.logerr("Failed to send pose 2")
            success = False
        else:
            rospy.loginfo("Waiting for pose 2 to finish...")

        self.arm.wait_for_action_end_or_abort()

        return success

    def verify(self) -> bool:
        return True
