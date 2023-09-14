#  GUI for debugging
<p float="center">
  <img src="images/gui/gui.jpg" width="600" />
</p>

To allow for intuitive and interactive debugging of the robot movements and poses, a GUI was developed. The GUI is written in Python and uses the Tkinter library.

The GUI allows the user to:
- Get the current pose of the robot in:
    - Base frame (x, y, z, roll, pitch, yaw)
    - Board frame (x, y, z, qx, qy, qz, qw) - The pose wrt to the board link which is attached to the blue button on the board indiciating the origin of the board frame
    - Joint angles
- Control the Gripper:
    - Open
    - Close
    - Set width (0.0 - 1.0)
- Move the arm to:
    - A pose in the base frame (x, y, z, roll, pitch, yaw)
    - A pose in the board frame (x, y, z, qx, qy, qz, qw)
    - A pose in the joint space (joint angles)
- Load the various pose lists from the config directory and allows to select a pose from the list and move the arm to it:
    - Joint angles: Poses specified in the joint space
    - Fixed transforms: Poses specified wrt board frame
    - Probe action: Poses used for the probe action task
    - Winding poses: Poses used for the winding task
    - BYOD poses: Poses used for the BYOD task

The GUI is modular and can be easily extended to add more pose lists and tasks. 

The main intention behind creating a GUI is to easily debug various tasks by verifying the poses and movements of the robot. This allows us to visit any pose from the lists and adjust them if necessary.

## Running the GUI

### Requirements
- Tkinter
- Pyperclip

### Initial Step

- Make sure the robot is connected to the computer and the robot is powered on.

- Make sure all the required yaml files are loaded into parameter server.

- Make sure the task_board_detector is running and the board_link is being published in tf.

### Running the GUI

```bash
roslaunch kinova_apps gui_kinova.launch
```

### Adding new pose lists

- Add the new poses in yaml file in the config directory.
- In the gui_kinova.py, follow the same format on how to load the yaml fiel and create a window.
- A callback method has to be defined for your pose list to handle what to do when some item is slected from the new list.
