#!/usr/bin/env python3

import rospy
from kinova_apps.transform_utils import TransformUtils
from utils.kinova_pose import KinovaPose
from geometry_msgs.msg import PoseStamped
from kortex_driver.srv import *
from kortex_driver.msg import *


class PubTest:
    def __init__(self):
        self.trasnform_utils = TransformUtils()
        self.pub = rospy.Publisher(
            "/my_gen3/pose_in_board", PoseStamped, queue_size=10
        )

        self.sub = rospy.Subscriber(
            "/my_gen3/base_feedback", BaseCyclic_Feedback, self.callback
        )

    def callback(self, data):
        x = data.base.tool_pose_x
        y = data.base.tool_pose_y
        z = data.base.tool_pose_z
        x_deg = data.base.tool_pose_theta_x
        y_deg = data.base.tool_pose_theta_y
        z_deg = data.base.tool_pose_theta_z

        kp = KinovaPose(x, y, z, x_deg, y_deg, z_deg)

        pose_in_bl = kp.to_pose_stamped()

        pose_in_board = self.trasnform_utils.transformed_pose_with_retries(
            pose_in_bl, "board_link"
        )

        self.pub.publish(pose_in_board)

        print("pose_in_board: ", pose_in_board)
        print("*" * 20)


if __name__ == "__main__":
    rospy.init_node("pose_in_board")
    PT = PubTest()
    rospy.spin()
