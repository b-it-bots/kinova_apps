#!/usr/bin/env python3
import tf
import rospy
import numpy as np
import math
from kortex_driver.msg import TwistCommand, CartesianReferenceFrame
import kortex_driver.msg

from kinova_apps.abstract_action import AbstractAction
from kinova_apps.full_arm_movement import FullArmMovement
from geometry_msgs.msg import PoseStamped, Quaternion, Twist, Vector3
from utils.transform_utils import TransformUtils
from utils.kinova_pose import KinovaPose, get_kinovapose_from_pose_stamped
from utils.force_measure import ForceMeasurmement


class SliderAction(AbstractAction):
    def __init__(self, arm: FullArmMovement, transform_utils: TransformUtils):
        super().__init__(arm, transform_utils)
        self.arm = arm
        self.fm = ForceMeasurmement()
        self.tf_utils = transform_utils
        self.listener = tf.TransformListener()
        self.slider_pose = PoseStamped()
        self.current_force_z = []

        self.cartesian_velocity_pub = rospy.Publisher(
            "/my_gen3/in/cartesian_velocity", TwistCommand, queue_size=1
        )
        self.base_feedback_sub = rospy.Subscriber(
            "/my_gen3/base_feedback",
            kortex_driver.msg.BaseCyclic_Feedback,
            self.base_feedback_cb,
        )

    def base_feedback_cb(self, msg):
        self.current_force_z.append(msg.base.tool_external_wrench_force_z)
        if len(self.current_force_z) > 25:
            self.current_force_z.pop(0)

    def pre_perceive(self) -> bool:
        print("in pre perceive")

        print("Getting slider pose")
        self.get_slider_pose()
        return True

    def act(self) -> bool:
        print("in act")

        rospy.loginfo(">> Opening gripper <<")
        if not self.arm.execute_gripper_command(0.55):
            return False

        rospy.loginfo(">> Moving arm to slider <<")
        if not self.move_arm_to_slider():
            return False

        rospy.loginfo(">> Approaching slider with caution <<")
        # success = self.arm.move_down_with_caution(distance=0.05, time=3.0, force_threshold=[4,4,4.5], retract_dist=0.01, tool_z_thresh=0.1095)
        # success = self.move_down_with_caution()
        board_height = rospy.get_param("/board_height", 0.1157)
        current_pose = self.arm.get_current_pose()
        current_pose.z = board_height + 0.003
        success = self.arm.send_cartesian_pose(current_pose)
        if not success:
            return False
        # if not self.approach_slider_with_caution():
        #    return False

        rospy.loginfo(">> Closing gripper <<")
        if not self.arm.execute_gripper_command(0.75):
            return False

        rospy.loginfo(">> Moving arm along the slider <<")
        if not self.move_arm_along_slider(direction="forward"):
            return False

        rospy.loginfo(">> Stopping arm <<")
        if not self.stop_arm():
            return False

        rospy.loginfo(">> Moving slider back  <<")
        if not self.move_arm_along_slider(direction="backward"):
            return False

        rospy.loginfo(">> Stopping arm <<")
        if not self.stop_arm():
            return False

        rospy.loginfo(">> open gripper <<")
        if not self.arm.execute_gripper_command(0.50):
            return False

        rospy.loginfo(">> Retract arm back <<")
        current_pose = self.arm.get_current_pose()
        current_pose.z += 0.03
        self.arm.send_cartesian_pose(current_pose)
        # if not self.retract_arm_back():
        #    return False

        rospy.loginfo(">> Stopping arm <<")
        if not self.stop_arm():
            return False

        rospy.loginfo(">> Process finished successfully <<")

        return True

    def verify(self) -> bool:
        print("in verify")

        ## can be verified by checking the current location of the EEF
        return True

    def do(self) -> bool:
        success = True

        success &= self.pre_perceive()
        success &= self.act()
        success &= self.verify()

        return success

    def tooltip_pose_callback(self, msg):
        self.tooltip_pose_z_with_base = msg.base.tool_pose_z

    def move_down_with_caution(self):
        self.current_force_z = []
        num_retries = 0
        loop_rate = rospy.Rate(10)
        force_z_diff_threshold = rospy.get_param("~force_z_diff_threshold", 4.0)
        stop = False

        while not rospy.is_shutdown():
            if len(self.current_force_z) < 20:
                num_retries += 1
                if num_retries > 100:
                    rospy.logerr("No force measurements received")
                    break
                loop_rate.sleep()
                continue
            msg = kortex_driver.msg.TwistCommand()
            msg.reference_frame = (
                kortex_driver.msg.CartesianReferenceFrame.CARTESIAN_REFERENCE_FRAME_TOOL
            )
            msg.twist.linear_z = 0.01
            if (
                abs(np.mean(self.current_force_z) - self.current_force_z[-1])
                > force_z_diff_threshold
            ):
                stop = True
                msg.twist.linear_z = 0.0
                msg.reference_frame = (
                    kortex_driver.msg.CartesianReferenceFrame.CARTESIAN_REFERENCE_FRAME_MIXED
                )  # publish 0 vel in mixed frame
            if self.arm.get_current_pose().z < 0.11:
                stop = True
                msg.twist.linear_z = 0.0
                msg.reference_frame = (
                    kortex_driver.msg.CartesianReferenceFrame.CARTESIAN_REFERENCE_FRAME_MIXED
                )  # publish 0 vel in mixed frame
            self.cartesian_velocity_pub.publish(msg)
            if stop:
                break
            loop_rate.sleep()
        current_pose = self.arm.get_current_pose()
        current_pose.z += 0.01
        self.arm.send_cartesian_pose(current_pose)
        return True

    def move_arm_to_slider(self):
        """
        Move arm to along slider in with velocity vector
        """

        offset = 0.1
        rospy.loginfo("slider pose below")
        print(self.slider_pose)
        slider_pose = self.rotate_Z_down(self.slider_pose)
        pose_to_send = get_kinovapose_from_pose_stamped(slider_pose)
        pose_to_send.z += (
            offset  # add 10 cm to the z axis and then approach the slider
        )

        success = self.arm.send_cartesian_pose(pose_to_send)
        if not success:
            return False
        return True

    def approach_slider_with_caution(self):
        """
        Approach slider with caution
        """

        offset = 0.05  # offset for the distance from tool frame to the tool tip
        rate_loop = rospy.Rate(10)
        self.fm.set_force_threshold(
            force=[4, 4, 3]
        )  # force in z increases to 4N when it is in contact with the board

        # enable force monitoring
        self.fm.disable_monitoring()
        self.fm.enable_monitoring()

        # calculate velocity
        distance = offset
        time = 3  # move 5 cm in 6 seconds
        velocity = distance / time

        # create twist command to move towards the slider
        approach_twist = TwistCommand()
        approach_twist.reference_frame = (
            CartesianReferenceFrame.CARTESIAN_REFERENCE_FRAME_TOOL
        )
        approach_twist.twist.linear_z = velocity

        height_limit_flat = False
        while not self.fm.force_limit_flag and not rospy.is_shutdown():
            # check for force limit flag and stop if it is true
            # check for the z axis of the tooltip and stop if it is less than 0.111(m) (the z axis of the slider) from base frame

            if self.fm.force_limit_flag:
                break
            if self.arm.get_current_pose().z < 0.1095:
                height_limit_flag = True
                break
            self.cartesian_velocity_pub.publish(approach_twist)
            rate_loop.sleep()

        success = self.stop_arm()
        if not success:
            return False

        self.fm.disable_monitoring()

        distance = 0.01
        time = 1  # move back 8 mm
        velocity = distance / time

        retract_twist = TwistCommand()
        retract_twist.reference_frame = (
            CartesianReferenceFrame.CARTESIAN_REFERENCE_FRAME_TOOL
        )
        retract_twist.twist.linear_z = -velocity
        if self.fm.force_limit_flag or height_limit_flag:
            self.cartesian_velocity_pub.publish(retract_twist)
            rospy.sleep(time)

        self.stop_arm()
        return True

    def move_arm_along_slider(self, direction: str = "None"):
        """
        Move arm along slider
        """

        if direction == "forward":
            dv = 1
            distance = 0.0255
        elif direction == "backward":
            dv = -1
            distance = 0.035
        elif direction == "None":
            print("No direction specified")
            return False
        else:
            print("Invalid direction")
            return False

        # calculate force in slider axis

        # #get current yaw angle
        # current_pose = self.arm.get_current_pose()
        # force_theta = current_pose.theta_z_deg

        # #calcuate mag in given angle
        # slider_force_threshold = 3 # N
        # theta = np.deg2rad(force_theta)

        # force_x = max( abs(slider_force_threshold * math.cos(theta)) , 2.5)
        # force_y = max( abs(slider_force_threshold * math.sin(theta)) , 2.5)

        # print("force_x: ", force_x)
        # print("force_y: ", force_y)

        success = self.arm.move_down_with_caution(
            distance=dv * distance,
            time=4,
            force_threshold=[10, 10, 10],
            approach_axis="x",
            retract=False,
            retract_dist=0.01,
        )
        if not success:
            return False

        rospy.loginfo(">> moved slider forward <<")
        rospy.loginfo(">> SLEEPING for 0.5 <<")

        rospy.sleep(0.5)
        return True

    def retract_arm_back(self):
        """
        Move arm back using twist command
        """
        retract_velocity = 0.07  # m/s
        retract_twist_cmd = TwistCommand()
        retract_twist_cmd.reference_frame = (
            CartesianReferenceFrame.CARTESIAN_REFERENCE_FRAME_TOOL
        )
        retract_twist_cmd.twist.linear_z = (
            -retract_velocity
        )  # neg velocity for upwards movement
        self.cartesian_velocity_pub.publish(retract_twist_cmd)
        rospy.sleep(1)
        return True

    def stop_arm(self):
        """
        Stop arm by sending zero velocity
        """

        velocity_vector = TwistCommand()
        velocity_vector.reference_frame = (
            CartesianReferenceFrame.CARTESIAN_REFERENCE_FRAME_MIXED
        )  # for proper joypad control
        self.cartesian_velocity_pub.publish(velocity_vector)
        return True

    def get_slider_pose(self):
        """
        Get slider pose
        """
        msg = PoseStamped()
        msg.header.frame_id = "slider_start_link"
        msg.header.stamp = rospy.Time(0)
        self.slider_pose = self.transform_utils.transformed_pose_with_retries(
            msg, "base_link"
        )

        return True

    def rotate_Z_down(self, msg: PoseStamped) -> PoseStamped:
        msg = msg
        msg.header.stamp = rospy.Time.now()

        # quaternion to euler for given pose
        e = list(
            tf.transformations.euler_from_quaternion(
                [
                    msg.pose.orientation.x,
                    msg.pose.orientation.y,
                    msg.pose.orientation.z,
                    msg.pose.orientation.w,
                ]
            )
        )

        rot_eluer = [
            math.pi,
            0.0,
            math.pi,
        ]  # rotate 180 degrees around x axis to make the z pointing (down) and the y will be in opposite direction than original and we
        # add 180 around z axis to set the orientation of camera facing the gui screen

        e[0] += rot_eluer[0]
        e[1] += rot_eluer[1]
        e[2] += rot_eluer[2]

        q = list(tf.transformations.quaternion_from_euler(e[0], e[1], e[2]))

        msg.pose.orientation = Quaternion(*q)
        msg = self.transform_utils.transformed_pose_with_retries(
            msg, "base_link"
        )
        return msg
