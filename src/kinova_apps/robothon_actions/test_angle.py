
import math
import numpy as np
import rospy
import time as tt
# angle = 0

# radius = 0.055
# arc_length = (2 * math.pi * radius) /4
# travel_time = 1
# angular_velocity_door = arc_length / travel_time
# # angular_velocity_door *= 10
# travel_time = 15

# prev_theta = 0

# for t in range(travel_time):
    
#     print("time==>: ", t, "s")

#     # angle += np.deg2rad(30/travel_time)

#     angle_feedback = np.rad2deg(angular_velocity_door * t)


#     # print("angluar velocity: ", angular_velocity_door, "rad/s")
#     # print("angle_feedback: ", angle_feedback)
#     # print("angle: ", angle)

#     angle = np.deg2rad(angle_feedback)

#     print("angle: ", angle)

#     # angular roataion calculation
#     z = radius * math.cos(angle)
#     y = radius * math.sin(angle)

#     theta = np.arctan2(z, y)
#     print("theta: ", theta)
#     current_theta = theta

#     #find the rate of change of theta
#     d_theta = current_theta - prev_theta
#     prev_theta = current_theta
#     print("d_theta: ", d_theta)

#     #angular velocity in x calculation
#     angular_velocity_x = d_theta 


#     # update step
#     linear_velocity_z = radius * angular_velocity_door * math.cos(angle)
#     linear_velocity_x = radius * angular_velocity_door * math.sin(angle) + (radius * (angular_velocity_door)**2)
#             # if(t > 5 and t < 10):



#     # print("linear_velocity_x: ", linear_velocity_x)
#     # print("linear_velocity_z: ", linear_velocity_z)
    
#     print("++++++++++++++++++++++++++++++"+"\n")
    
#     print("++++++++++++++++++++++++++++++"+"\n")


#     # if(t > 5 and t < 10):


def garbage():


    angle = 0

    radius = 0.07

    radius_y = 0.07
    radius_z = 0.15

    arc_length = (2 * math.pi * radius) /4
    travel_time = 1
    angular_velocity_door = arc_length / travel_time

    # angular_velocity_door *= 10
    travel_time = 15

    prev_theta = 0

    for t in range(travel_time):
        
        print("time==>: ", t, "s")
        angle_feedback = np.rad2deg(angular_velocity_door * t)

        print("angluar velocity of the door : ", angular_velocity_door, "rad/s")
        print("angle_feedback of the motion : ", angle_feedback)
        angle = np.deg2rad(angle_feedback)


        #ellipse motion

        velocity_door_ellipse = angular_velocity_door * math.sqrt( radius_z**2 * math.cos(angle)**2 +
                                                                    radius_y**2 * math.sin(angle)**2)

        omega_ =  (angular_velocity_door /  math.sqrt( radius_z**2 * math.cos(angle)**2 +
                                                    radius_y**2 * math.sin(angle)**2))


        # y and z calculation
        y = omega_ * radius_y * math.cos(angle)
        z = omega_ * radius_z * math.sin(angle)

        #theta calculation
        theta = omega_ * t  


        # velocity in y and z calculation
        vy = -radius_y * math.sin(angle) * velocity_door_ellipse
        vz = radius_z * math.cos(angle) * velocity_door_ellipse

        # assignment step

        linear_velocity_y = vy
        linear_velocity_z = vz



        # # angular roataion calculation
        # z = radius * math.cos(angle)
        # y = radius * math.sin(angle)

        # theta = np.arctan2(z, y)
        # print("theta of perpendicular vector: ", theta)
        # current_theta = theta

        # #find the rate of change of theta
        # d_theta = current_theta - prev_theta
        # prev_theta = current_theta
        # print("rate of change in angle: ", d_theta)

        # #angular velocity in x calculation
        # angular_velocity_x = d_theta /4 

        # # update step
        # linear_velocity_z = radius * angular_velocity_door * math.cos(angle)
        # linear_velocity_y = radius * angular_velocity_door * math.sin(angle) + (radius * (angular_velocity_door)**2)

        # #conditions 
        # if t == 0:
        #     angular_velocity_x = 0
        
        print("++++++++++++++++++++++++++++++")
        # printing step 
        print("linear_velocity_y: ", linear_velocity_y, "m/s")
        print("linear_velocity_z: ", linear_velocity_z, "m/s")
        # print("angular_velocity_x: ", angular_velocity_x, "rad/s")
        
        print("++++++++++++++++++++++++++++++"+"\n")


def ellipse_test():

    angle = 0
    radius = 0.07
    radius_y = 0.07
    radius_z = 0.15
    arc_length = (2 * math.pi * radius) /4
    travel_time = 1
    angular_velocity_door = arc_length / travel_time
    angular_velocity_door *= 10
    travel_time = 15

    for t in range(travel_time):
        
        print("time==>: ", t, "s")
        # angle_feedback = np.rad2deg(angular_velocity_door * t)

        print("angluar velocity of the door : ", angular_velocity_door, "rad/s")
        # print("angle_feedback of the motion : ", angle_feedback)
        # angle = np.deg2rad(angle_feedback)

        angle += np.deg2rad(90/travel_time)

        #ellipse motion
        velocity_door_ellipse = angular_velocity_door * math.sqrt( radius_z**2 * math.cos(angle)**2 +
                                                                    radius_y**2 * math.sin(angle)**2)

        omega_ =  (angular_velocity_door /  math.sqrt( radius_z**2 * math.cos(angle)**2 +
                                                    radius_y**2 * math.sin(angle)**2))


        # y and z calculation
        y = omega_ * radius_y * math.cos(angle)
        z = omega_ * radius_z * math.sin(angle)

        #theta calculation
        theta = omega_ * t  


        # velocity in y and z calculation
        vy = -radius_y * math.sin(angle) * velocity_door_ellipse
        vz = radius_z * math.cos(angle) * velocity_door_ellipse

        # assignment step

        linear_velocity_y = vy
        linear_velocity_z = vz


        print("++++++++++++++++++++++++++++++")
        # printing step 
        print("linear_velocity_y: ", linear_velocity_y, "m/s")
        print("linear_velocity_z: ", linear_velocity_z, "m/s")

        print("++++++++++++++++++++++++++++++"+"\n")

def angle():

    # force = [1,4]
    force_theta = 26
    theta = np.deg2rad(force_theta)
    force_mag = 3
    force_x = force_mag * math.cos(theta)
    force_y = force_mag * math.sin(theta)

    print("force_x: ", force_x)
    print("force_y: ", force_y)

def open_door(self):

    """
    this functions has different methods tested to open  the door  

    """

    # get magnet pose from param server
    magnet_pose = rospy.get_param("~magnet_pose")
    magnet_kinova_pose = get_kinovapose_from_list(magnet_pose)

    # rotate the yaw by 180 degrees
    magnet_kinova_pose.theta_z_deg += 180.0

    # send magnet pose to the arm
    print("[probe_action] moving to magnet position")
    success = self.arm.send_cartesian_pose(magnet_kinova_pose)

    if not success:
        rospy.logerr("Failed to move to the magnet position")
        return False
    
    print("[probe_action] reached magnet position")

    # close the gripper
    success = self.arm.execute_gripper_command(0.7)

    if not success:
        rospy.logerr("Failed to close the gripper")
        return False
    
    # move up a bit
    magnet_kinova_pose.z += 0.3

    success = self.arm.send_cartesian_pose(magnet_kinova_pose)

    if not success:
        rospy.logerr("Failed to move up the probe")
        return False
    

#### door opening without magnet


    # go to the door knob position
    # get the door knob position from tf
    msg = PoseStamped()
    msg.header.frame_id = "door_knob_link"
    msg.header.stamp = rospy.Time(0)
    # msg.pose.position.x -= 0.015

    door_knob_pose = self.transform_utils.transformed_pose_with_retries(msg, "base_link", execute_arm=True, offset=[0.0, 0.0, -math.pi/2])

    # convert the door knob pose to a kinova pose
    door_knob_kinova_pose = get_kinovapose_from_pose_stamped(door_knob_pose)

    # move up a bit
    door_knob_kinova_pose.z += 0.022 + 0.05 # adding 5 cm for approach

    # send the door knob pose to the arm
    print("[probe_action] moving to door knob position")
    print("[probe_action] door knob pose: {}".format(door_knob_kinova_pose))

    success = self.arm.send_cartesian_pose(door_knob_kinova_pose)
    # return False
    rospy.sleep(1.0) # wait for the arm to settle for proper force sensing

    if not success:
        rospy.logerr("Failed to move to the door knob position")
        return False



    # velocity mode to approach the door knob
    success = self.arm.move_down_with_caution(velocity=0.01, force_threshold=[4,4,2.0])

    if not success:
        rospy.logerr("Failed to move down the arm")
        return False
    print("[probe_action] reached door knob position")



    # close the gripper
    success = self.arm.execute_gripper_command(0.75) # 0.75 closed 
    if not success:
        rospy.logerr("Failed to open the gripper")
        return False
    

    # # linear_vel_z = 0.0025
    # vel = 0.01
    # arc_radius = 0.07
    # angle = 45
    # angular_velocity = vel/arc_radius

    # angular_velocity *= 0.05 # 20% of the linear velocity
    # # angle = 25.0
    # # # w.r.t board link
    # # linear_vel_x = vel*math.cos(math.radians(angle))
    # # linear_vel_z = vel*math.sin(math.radians(angle))


    door_open_twist = kortex_driver.msg.TwistCommand()
    door_open_twist.reference_frame = CartesianReferenceFrame.CARTESIAN_REFERENCE_FRAME_TOOL


    # linear_velocity_x = abs(vel*math.cos(math.degrees(angle)))
    # linear_velocity_z = abs(vel*math.sin(math.degrees(angle)))
    # angular_velocity_y = abs(angular_velocity)

    # linear_velocity_x *= 0.95
    # linear_velocity_z *= 1.1

    # switch = -1
    # kill_z = 0

    angle = 0
    radius = 0.07
    radius_y = 0.068
    radius_z = 0.065
    arc_length = (2 * math.pi * radius) /4
    travel_time = 1
    angular_velocity_door = arc_length / travel_time
    angular_velocity_door *= 10
    travel_time = 15

    for t in range(travel_time):
        
        print("time==>: ", t, "s")
        # angle_feedback = np.rad2deg(angular_velocity_door * t)

        print("angluar velocity of the door : ", angular_velocity_door, "rad/s")
        # print("angle_feedback of the motion : ", angle_feedback)
        # angle = np.deg2rad(angle_feedback)

        angle += np.deg2rad(90/travel_time)

        #ellipse motion

        velocity_door_ellipse = angular_velocity_door * math.sqrt( radius_z**2 * math.cos(angle)**2 +
                                                                    radius_y**2 * math.sin(angle)**2)

        omega_ =  (angular_velocity_door /  math.sqrt( radius_z**2 * math.cos(angle)**2 +
                                                    radius_y**2 * math.sin(angle)**2))


        # y and z calculation
        y = omega_ * radius_y * math.cos(angle)
        z = omega_ * radius_z * math.sin(angle)

        #theta calculation
        theta = omega_ * t  


        # velocity in y and z calculation
        vy = -radius_y * math.sin(angle) * velocity_door_ellipse
        vz = radius_z * math.cos(angle) * velocity_door_ellipse

        # assignment step

        linear_velocity_y = vy
        linear_velocity_z = vz


        print("++++++++++++++++++++++++++++++")
        # printing step 
        print("linear_velocity_y: ", linear_velocity_y, "m/s")
        print("linear_velocity_z: ", linear_velocity_z, "m/s")

        print("++++++++++++++++++++++++++++++"+"\n")

        # publishing step
        door_open_twist.twist.linear_y = -abs(linear_velocity_y) # negative sign because the tool frame is facing the downward direction
        door_open_twist.twist.linear_z = -abs(linear_velocity_z) 
        # door_open_twist.twist.angular_x = angular_velocity_x
        self.cart_vel_pub.publish(door_open_twist)

        rospy.sleep(1)

    msg = kortex_driver.msg.TwistCommand()

    msg.twist.linear_z = 0.0
    msg.twist.linear_x = 0.0
    msg.twist.linear_y = 0.0
    msg.twist.angular_x = 0.0
    msg.twist.angular_y = 0.0
    msg.twist.angular_z = 0.0

    self.cart_vel_pub.publish(msg)

    self.arm.clear_faults()
    self.arm.subscribe_to_a_robot_notification()

### door opening without magnet

    # get current pose of the arm
    current_pose = self.arm.get_current_pose()

    # go up by 5cm
    current_pose.z += 0.1

    success = self.arm.send_cartesian_pose(current_pose)

    if not success:
        rospy.logerr("Failed to move up the probe")
        return False
    
    # go to the magnet position
    print("[probe_action] moving to magnet position")
    magnet_pose = rospy.get_param("~magnet_pose")
    magnet_kinova_pose = get_kinovapose_from_list(magnet_pose)
    magnet_kinova_pose.theta_z_deg += 180.0
    success = self.arm.send_cartesian_pose(magnet_kinova_pose)

    if not success:
        rospy.logerr("Failed to move to the magnet position")
        return False

    # open the gripper
    success = self.arm.execute_gripper_command(0.35)

    if not success:
        rospy.logerr("Failed to open the gripper")
        return False
    
    # move up a bit
    magnet_kinova_pose.z += 0.3

    success = self.arm.send_cartesian_pose(magnet_kinova_pose)

    if not success:
        rospy.logerr("Failed to move up the probe")
        return False

    print("[probe_action] target reached")

def push_door(self):  # push door is not cuurently used
        # move arm above door knob position 

    msg = PoseStamped()
    msg.header.frame_id = "door_knob_link"
    msg.header.stamp = rospy.Time(0)
    msg = PoseStamped()
    msg.header.frame_id = "door_knob_link"
    msg.header.stamp = rospy.Time(0)

    msg.pose.position.x -= 0.075
    msg.pose.position.z += 0.30
    msg.pose.position.x -= 0.075
    msg.pose.position.z += 0.30

    door_knob_pose = self.transform_utils.transformed_pose_with_retries(msg, "base_link", execute_arm=True)
    door_knob_pose = self.transform_utils.transformed_pose_with_retries(msg, "base_link", execute_arm=True)

    print("Door knob pose:  ")
    print(door_knob_pose)
    self.door_knob_pose_pub.publish(door_knob_pose)
    print("Door knob pose:  ")
    print(door_knob_pose)
    self.door_knob_pose_pub.publish(door_knob_pose)

    rospy.sleep(5)
    rospy.sleep(5)

    success = self.arm.execute_gripper_command(1.0)
    success = self.arm.execute_gripper_command(1.0)

    if not success:
        rospy.logerr("Failed to move up the probe")
        return False
    
    # go to the magnet position
    print("[probe_action] moving to magnet position")
    magnet_pose = rospy.get_param("~magnet_pose")
    magnet_kinova_pose = get_kinovapose_from_list(magnet_pose)
    magnet_kinova_pose.theta_z_deg += 180.0
    success = self.arm.send_cartesian_pose(magnet_kinova_pose)

    if not success:
        rospy.logerr("Failed to move to the magnet position")
        return False

    # open the gripper
    success = self.arm.execute_gripper_command(0.35)

    if not success:
        rospy.logerr("Failed to open the gripper")
        return False
    
    # move up a bit
    magnet_kinova_pose.z += 0.3

    success = self.arm.send_cartesian_pose(magnet_kinova_pose)

    if not success:
        rospy.logerr("Failed to move up the probe")
        return False

    print("[probe_action] target reached")    

def time_calculator():

    time = 5
    time_now = tt.time()

    while True:
        if abs(time_now - tt.time()) >= time:
            print("time is up")
            break
def appending():

    listi = [[5]]
    listi.append(([1],[2]))

    print(listi)
if __name__ == "__main__":

    appending()