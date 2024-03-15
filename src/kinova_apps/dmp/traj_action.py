import sys

import numpy as np
import rospy
import time
from kortex_driver.srv import *
from kortex_driver.msg import *

from kinova_apps.abstract_action import AbstractAction

from kinova_apps.full_arm_movement import FullArmMovement
from utils.transform_utils import TransformUtils

from utils.kinova_pose import (
    KinovaPose,
    get_kinovapose_from_list,
    get_kinovapose_from_pose_stamped,
)

from copy import deepcopy

from pydmps.dmp_discrete import DMPs_discrete
from pydmps.utils.plot_utils import plot_2d


class DMPTrajectoryAction(AbstractAction):
    def __init__(self, arm: FullArmMovement, transform_utils: TransformUtils):
        super().__init__(arm, transform_utils)

        rospy.loginfo("DMP Trajectory Action")

        self.dmp_discrete = DMPs_discrete(
            n_dmps=6, load_model=True, model_name="trained_model_2"
        )

        self.start_pose = [
            0.36287772,
            0.2259503,
            0.18019469,
            -3.04193887,
            0.04809582,
            -1.50937299,
        ]

        self.goal_pose = [0.298367, -0.3168, 0.1939, -3.1410348, -0.020932, -1.5715458]

    def pre_perceive(self) -> bool:

        # convert start pose last 3 values from radians to degrees
        self.start_pose[3] = 180 * self.start_pose[3] / 3.1415
        self.start_pose[4] = 180 * self.start_pose[4] / 3.1415
        self.start_pose[5] = 180 * self.start_pose[5] / 3.1415

        # go to start pose
        kpose = get_kinovapose_from_list(self.start_pose)
        rospy.loginfo("Going to start pose")
        self.arm.send_constrained_pose(kpose)

        return True

    def act(self) -> bool:
        success = True

        # Prepare and send pose 1
        start_pose = self.arm.get_current_pose()
        # goal_pose = deepcopy(start_pose)
        goal_pose = get_kinovapose_from_list(self.goal_pose)
        # goal_pose.y += 0.6

        current_pose = start_pose.to_pose_stamped()

        print(f"sp: {start_pose}")
        print(f"gp: {goal_pose}")

        # transform to camera_placeholder frame
        # current_pose = self.transform_utils.transform_pose(
        #     current_pose, "camera_placeholder"
        # )

        current_pose = self.transform_utils.convert_to_euler(current_pose)

        # goal_pose = self.transform_utils.transform_pose(
        #     goal_pose.to_pose_stamped(), "camera_placeholder"
        # )

        goal_pose = goal_pose.to_pose_stamped()

        goal_pose = self.transform_utils.convert_to_euler(goal_pose)

        self.dmp_discrete.set_y0(current_pose)
        self.dmp_discrete.set_goal(goal_pose)

        print("*" * 50)
        print(f"cp: {current_pose}")
        print(f"gp: {goal_pose}")
        print("*" * 50)

        dat = np.array([])
        dat = np.append(dat, current_pose)

        x_track = 0.99
        rospy.loginfo("----> fire up the DMP")
        counter = 0
        while x_track > 0.001 and rospy.is_shutdown() is False and counter < 100:
            counter += 1
            y_track, dy_track, ddy_track, f_track, x_track = self.dmp_discrete.rollout(
                current_pose, 1
            )
            # current_pose = deepcopy(y_track)
            dat = np.vstack((dat, y_track))

            # target_pose = get_kinovapose_from_list(y_track[:6]).to_pose_stamped("camera_placeholder")
            target_pose = get_kinovapose_from_list(y_track[:6])

            # # transform to camera_placeholder frame
            # target_pose = self.transform_utils.transform_pose(
            #     target_pose, "base_link"
            # )

            # target_pose = get_kinovapose_from_pose_stamped(target_pose)

            target_pose.theta_x_deg = start_pose.theta_x_deg
            target_pose.theta_y_deg = start_pose.theta_y_deg
            target_pose.theta_z_deg = start_pose.theta_z_deg

            print(f"yt: {y_track[:6]}")
            print(f"tp: {target_pose}")

            success &= self.arm.send_constrained_pose(target_pose)

            current_pose = self.arm.get_current_pose().to_pose_stamped()
            # current_pose = self.transform_utils.transform_pose(
            #     current_pose, "camera_placeholder"
            # )

            current_pose = self.transform_utils.convert_to_euler(current_pose)
            print(f"cp: {current_pose}")

        plot_2d(dat)

        return success

    def verify(self) -> bool:
        return True
