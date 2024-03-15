# Kinova Apps
This repository contains the applications and documentation developed for the Kinova Gen3 Arm. 



# Quick Start Guide
## Dependencies
We use Ubuntu 20.04 / Pop! OS 20.04 as the operating system, with ROS Noetic for interprocess communication.
### ROS dependencies
- [kortex](https://github.com/Kinovarobotics/kortex)
- [ros_kortex](https://github.com/Kinovarobotics/ros_kortex)
- [ros_kortex_vision](https://github.com/Kinovarobotics/ros_kortex_vision)
- [cv_bridge](http://wiki.ros.org/cv_bridge)
- ros-noetic-pcl-ros
- ros-noetic-perception-pcl

## Setup the robot

1. Connect an ethernet cable from the robot to the computer
2. Open Settings -> Network -> Wired (click on setting icon) -> Go to IPv4 -> Click on "Manual" IPv4 Method -> Under Addresses, in Address, type `192.168.1.110` | in Netmask, type `255.255.255.0` -> Apply
3. On the browser (Chrome), type URL `192.168.1.12` (assuming the address of the robot is configured to 192.168.1.12)
4. Username: admin, Password: <type-password-here>

## Update launch file

1. roscd kortex_driver/launch
2. Open kortex_driver.launch
3. Change the IP address to `192.168.1.12`
4. Change password to the password you set
5. Add default gripper, it should look like this (line 13) `<arg name="gripper" default="robotiq_2f_85" if="$(eval arg('arm') == 'gen3')"/>`
6. Change the IP address to `192.168.1.12` (line 3) in `kinova_vision_rgbd.launch` file

## Launch

- Launch the necessary nodes to bringup the arm and the camera.

- Run each of the following in separate terminals:

  ```
  roscore
  roslaunch kortex_driver kortex_driver.launch
  roslaunch kinova_vision kinova_vision_rgbd.launch
  ```

  Visualize debug images and TF tree in RViz using [config/kinova.rviz](config/robothon.rviz)

  
## Apps

### DMP

- Create workspace
  
    ```bash
    mkdir -p ~/dmp_ws/src
    cd ~/dmp_ws/src
    ```

- Clone necessary repositories

  ```bash
  # ros-kortex
  git clone https://github.com/Kinovarobotics/ros_kortex.git

  # kinova_apps
  git@github.com:b-it-bots/kinova_apps.git -b dmp
  ```

- Clone the `pydmps` repository
  
  ```bash
  git clone https://github.com/Learning-in-robotics/pydmps.git -b deep-dmp
  ```

- Build the workspace

  ```bash
  cd ~/dmp_ws
  catkin build
  ```

- Install `pydmps` package
  ```bash
  cd ~/dmp_ws/src/pydmps
  pip3 install -e .
  ```
  
- Source the workspace

  ```bash
  source ~/dmp_ws/devel/setup.bash
  ```

- Launch the kinova arm in gazebo with the robotiq gripper

  ```bash
  roslaunch kortex_gazebo spawn_kortex_robot.launch gripper:=robotiq_2f_85
  ```

- Run the DMP test
  ```bash
  roslaunch kinova_apps dmp_traj_sim.launch
  ```
