#!/usr/bin/env python3
"""
This module contains a gui component for moving kinova arm

"""
# -*- encoding: utf-8 -*-

import math
import threading

import tkinter as tk
from tkinter import ttk
from tkinter import messagebox

import rospy

from typing import List, Tuple, Any, Optional

from kinova_apps.full_arm_movement import FullArmMovement
from utils.transform_utils import TransformUtils

from utils.kinova_pose import (
    get_kinovapose_from_list,
    get_kinovapose_from_pose_stamped,
)

from kortex_driver.srv import *
from kortex_driver.msg import *

from sensor_msgs.msg import JointState
from geometry_msgs.msg import PoseStamped

from typing import List, Optional


class Item:
    """A class that represents an item with a name and a value."""

    def __init__(self, name: str, value: List):
        """
        Initializes an instance of the Item class with a given name and value.

        Args:
            name (str): The name of the item.
            value (List): The value of the item.
        """
        self.name = name
        self.value = value

    def __str__(self):
        return f"Item(name={self.name}, value={self.value})"


class ItemList:
    """A class that represents a list of items and provides methods to interact with the list."""

    def __init__(self):
        """
        Initializes an instance of the ItemList class with an empty list of items.
        """
        self.items: List[Item] = []

    def add_item(self, item: Item) -> None:
        """
        Adds an Item object to the item list.

        Args:
            item (Item): The Item object to be added to the item list.
        """
        self.items.append(item)

    def add_items(self, items: List[Tuple[str, Any]]) -> None:
        """
        Adds a list of Item objects to the item list.

        Args:
            items (list): A list of Item objects to be added to the item list.
        """
        for name, value in items:
            item = Item(name, value)
            self.items.append(item)

    def get_item_by_name(self, name: str) -> Optional[Item]:
        """
        Retrieves an Item object from the item list by name.

        Args:
            name (str): The name of the item to be retrieved.

        Returns:
            Item or None: The Item object with the given name, or None if not found.
        """
        for item in self.items:
            if item.name == name:
                return item
        return None

    def get_all_items(self) -> List[Item]:
        """
        Retrieves all Item objects from the item list.

        Returns:
            list: A list of all Item objects in the item list.
        """
        return self.items

    def __iter__(self):
        """
        Returns an iterator over the items in the item list.

        Returns:
            iterator: An iterator over the items in the item list.
        """
        return iter(self.items)


class ListWindow(tk.Frame):
    def __init__(self, master: tk.Frame, name, items: ItemList, callback):
        super().__init__(master)
        self.master = master
        self.items = items
        self.callback = callback

        # Set listbox title
        self.title = ttk.Label(self, text=name)
        self.title.pack(padx=10, pady=10)

        # Create listbox with a scrollbar
        self.listbox = tk.Listbox(self, width=30)
        self.listbox.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        self.scrollbar = ttk.Scrollbar(self.listbox)
        self.listbox.config(yscrollcommand=self.scrollbar.set)
        self.scrollbar.config(command=self.listbox.yview)
        self.scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

        # Add items to listbox
        for item in self.items:
            self.listbox.insert(tk.END, item.name)

        # Create button with modern style
        self.button = ttk.Button(self, text="Go", command=self._on_button_click)
        self.button.pack(padx=5, pady=5)

        # Set listbox style
        style = ttk.Style()
        style.configure(
            "TListbox",
            font=("Helvetica", 12),
            background="#f0f0f0",
            foreground="#000000",
            borderwidth=0,
            highlightthickness=0,
        )
        style.map(
            "TListbox",
            background=[("selected", "#0078d7")],
            foreground=[("selected", "#ffffff")],
        )

        # Set button style
        style.configure("TButton", font=("Helvetica", 12))
        style.map(
            "TButton",
            background=[("active", "#0056b3"), ("selected", "#0056b3")],
            foreground=[("active", "#ffffff"), ("selected", "#ffffff")],
        )

    def _on_button_click(self):
        # get the current selection
        cs = self.listbox.curselection()

        if cs == ():
            pass
        else:
            # Get selected item from listbox
            selected_item_name = self.listbox.get(cs)

            # Call the callback function with the selected item as argument
            self.callback(self.items.get_item_by_name(selected_item_name))


class GuiKinova(object):
    """
    A class that represents a gui component for interacting with kinova arm.
    """

    def __init__(self):
        # get parameter with namespaced
        self.joint_angles = rospy.get_param("joint_angles", None)
        # self.probe_action_poses = rospy.get_param("~probe_action_poses", None)
        # self.wind_cable_poses = rospy.get_param("~wind_poses", None)
        # self.byod_poses = rospy.get_param("~byod_poses", None)
        # self.fixed_transforms = rospy.get_param("~fixed_transforms")

        self.lists = [self.joint_angles]

        self.arm = FullArmMovement()
        self.transform_utils = TransformUtils()

        # clear faults
        rospy.loginfo("Clearing faults...")
        self.arm.clear_faults()
        rospy.sleep(0.5)
        self.arm.subscribe_to_a_robot_notification()

        self.master = tk.Tk()  # Updated to use tk instead of Tkinter
        self.master.title("Kinova Arm GUI")
        self.master.geometry("1500x2000")

    def render_lists(self, noetbook: ttk.Notebook):
        # keep 3 lists in a frame
        frame = ttk.Frame(noetbook)

        # joint angles
        joint_angles = [
            (joint_angle, self.joint_angles[joint_angle])
            for joint_angle in self.joint_angles.keys()
        ]

        joint_anlges_list = ItemList()
        joint_anlges_list.add_items(joint_angles)

        joint_angles_window = ListWindow(
            frame, "joint angles", joint_anlges_list, self.joint_angles_cb
        )
        joint_angles_window.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        # # trajectories
        # trajectories = [(trajectory, self.trajectories[trajectory]) for trajectory in self.trajectories.keys()]

        # trajectories_list = ItemList()
        # trajectories_list.add_items(trajectories)

        # trajectories_window = ListWindow(frame, 'trajectories', trajectories_list, self.trajectories_cb)
        # trajectories_window.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        # fixed transforms
        # fixed_transform_link_names = [
        #     (link_name.replace("board_to_", "") + "_link", "")
        #     for link_name in self.fixed_transforms
        # ]

        # fixed_transforms_list = ItemList()
        # fixed_transforms_list.add_items(fixed_transform_link_names)

        # fixed_transforms_window = ListWindow(
        #     frame,
        #     "fixed transforms",
        #     fixed_transforms_list,
        #     self.fixed_transforms_cb,
        # )
        # fixed_transforms_window.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        # wind cable poses
        # wind_cable_poses = [
        #     (pose, self.wind_cable_poses[pose])
        #     for pose in self.wind_cable_poses
        # ]

        # wind_cable_poses_list = ItemList()
        # wind_cable_poses_list.add_items(wind_cable_poses)

        # wind_cable_poses_window = ListWindow(
        #     frame,
        #     "winding poses",
        #     wind_cable_poses_list,
        #     self.wind_cable_poses_cb,
        # )
        # wind_cable_poses_window.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        # probe action poses
        # probe_action_poses = [
        #     (pose, get_kinovapose_from_list(self.probe_action_poses[pose]))
        #     for pose in self.probe_action_poses.keys()
        # ]

        # probe_action_poses_list = ItemList()
        # probe_action_poses_list.add_items(probe_action_poses)

        # probe_action_poses_window = ListWindow(
        #     frame,
        #     "probe action poses",
        #     probe_action_poses_list,
        #     self.probe_action_poses_cb,
        # )
        # probe_action_poses_window.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        # second frame
        frame2 = ttk.Frame(noetbook)

        # byod poses
        # byod_poses = []
        # for key, i in zip(self.byod_poses.keys(), self.byod_poses.values()):
        #     byod_poses.append((key, get_kinovapose_from_list(list(i.values()))))

        # byod_poses_list = ItemList()
        # byod_poses_list.add_items(byod_poses)

        # byod_poses_window = ListWindow(
        #     frame2, "byod poses", byod_poses_list, self.byod_poses_cb
        # )
        # byod_poses_window.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        # add frames to notebook
        noetbook.add(frame, text="frame1")
        noetbook.add(frame2, text="frame2")

    def fixed_transforms_cb(self, item: Item):
        link_name = item.name

        self.arm.clear_faults()
        rospy.sleep(0.5)

        # get transform from fixed transform dictionary
        msg = PoseStamped()
        msg.header.frame_id = link_name
        msg.header.stamp = rospy.Time(0)

        msg_bl = self.transform_utils.transformed_pose_with_retries(
            msg, "base_link", execute_arm=True, offset=[0.0, 0.0, -math.pi / 2]
        )

        if msg_bl is None:
            rospy.logerr(
                "Failed to get transform from fixed transform dictionary"
            )
            messagebox.showerror(
                "Fixed Transform",
                "Failed to get transform from. Please check TF tree",
            )

        # send transform to robot
        kp = get_kinovapose_from_pose_stamped(msg_bl)
        kp.z += 0.1
        if not self.arm.send_cartesian_pose(kp):
            rospy.logerr("Failed to send transform to robot")

        # display popup window to inform user
        messagebox.showinfo(
            "Fixed Transform",
            "Fixed Transform sent to robot.\n Added 10cm +z offset for safety",
        )

    def probe_action_poses_cb(self, item: Item):
        pose = item.value
        success = self.arm.send_cartesian_pose(pose)

        # display popup window to inform user
        messagebox.showinfo(
            "Probe Action Pose", "Probe Action Pose sent to robot"
        )

    def byod_poses_cb(self, item: Item):
        kp = item.value
        self.arm.clear_faults()
        rospy.sleep(0.5)
        if not self.arm.send_cartesian_pose(kp):
            rospy.logerr("Failed to send byod pose to robot")

        # display popup window to inform user
        messagebox.showinfo("Byod Pose", "Byod Pose sent to robot")

    def joint_angles_cb(self, item: Item):
        self.arm.clear_faults()
        rospy.sleep(0.5)

        if not self.arm.send_joint_angles(item.value):
            rospy.logerr("Failed to send joint angles to robot")

        messagebox.showinfo("Joint Angles", "Joint Angles sent to robot")

    def trajectories_cb(self, item: Item):
        print("Trajectories button ", item)
        pass

    def wind_cable_poses_cb(self, item: Item):
        self.arm.clear_faults()
        rospy.sleep(0.5)

        msg = PoseStamped()
        msg.header.frame_id = "board_link"
        msg.pose.position.x = item.value[0]
        msg.pose.position.y = item.value[1]
        msg.pose.position.z = item.value[2]
        msg.pose.orientation.x = item.value[3]
        msg.pose.orientation.y = item.value[4]
        msg.pose.orientation.z = item.value[5]
        msg.pose.orientation.w = item.value[6]

        # convert to base_link
        msg = self.transform_utils.transformed_pose_with_retries(
            msg, "base_link"
        )

        kp = get_kinovapose_from_pose_stamped(msg)

        if not self.arm.send_cartesian_pose(kp):
            rospy.logerr("Failed to send wind cable pose to robot")

        # display popup window to inform user
        messagebox.showinfo("Wind Cable Pose", "Wind Cable Pose sent to robot")

    def gripper_command_window(self, frame: tk.Frame):
        # create a heading for the frame
        self.heading = tk.Label(
            frame, text="Gripper Command", font=("Helvetica", 14)
        )
        self.heading.grid(row=0, column=0, padx=10, pady=10, sticky="w")

        self.open_button = tk.Button(
            frame, text="Open", command=lambda: self.gripper_cb(0.35)
        )

        self.close_button = tk.Button(
            frame, text="Close", command=lambda: self.gripper_cb(1.0)
        )

        self.gripper_value = tk.DoubleVar()
        self.gripper_value.set(0.35)

        self.gripper_value_entry = tk.Entry(
            frame, textvariable=self.gripper_value
        )

        self.gripper_value_button = tk.Button(
            frame,
            text="Send",
            command=lambda: self.gripper_cb(self.gripper_value.get()),
        )

        # add all the widgets in one row
        # make the widgets move based on the window size
        self.open_button.grid(row=1, column=0, padx=10, pady=10, sticky="nsew")
        self.close_button.grid(row=1, column=1, padx=10, pady=10, sticky="nsew")
        self.gripper_value_entry.grid(
            row=1, column=2, padx=10, pady=10, sticky="nsew"
        )
        self.gripper_value_button.grid(
            row=1, column=3, padx=10, pady=10, sticky="nsew"
        )

        # configure column widths to be equal
        frame.grid_columnconfigure(0, weight=1)
        frame.grid_columnconfigure(1, weight=1)
        frame.grid_columnconfigure(2, weight=1)
        frame.grid_columnconfigure(3, weight=1)

    def gripper_cb(self, value):
        self.arm.execute_gripper_command(value)

    def current_pose_window(self, frame: tk.Frame):
        # Create a heading for the frame
        self.heading = tk.Label(
            frame, text="Current Tool Tip Pose", font=("Helvetica", 14)
        )
        self.heading.grid(row=0, column=0, padx=10, pady=10)

        # Create a text box to display the current tool tip pose in base frame
        self.base_frame_label = tk.Label(
            frame, text="Base Frame", font=("Helvetica", 10)
        )
        self.base_frame_label.grid(
            row=1, column=0, padx=10, pady=10, sticky="w"
        )

        self.base_frame_text = tk.Text(
            frame, height=3, width=30, font=("Helvetica", 10)
        )
        self.base_frame_text.grid(
            row=2, column=0, padx=10, pady=10, sticky="ew"
        )
        # self.base_frame_text.config(state='disabled')
        self.base_frame_text.configure()

        # Create a copy button for the base frame text
        self.copy_base_frame_button = ttk.Button(
            frame,
            text="Copy",
            command=lambda: self.copy_to_clipboard(
                self.base_frame_text.get("1.0", "end-1c")
            ),
        )
        self.copy_base_frame_button.grid(row=2, column=1, padx=10, pady=10)

        # create a goto button
        self.goto_base_frame_button = ttk.Button(
            frame,
            text="Goto",
            command=lambda: self.goto_base_frame_cb(
                self.base_frame_text.get("1.0", "end-1c")
            ),
        )
        self.goto_base_frame_button.grid(row=2, column=2, padx=10, pady=10)

        # Create a text box to display the current tool tip pose in board frame
        self.board_frame_label = tk.Label(
            frame, text="Board Frame", font=("Helvetica", 10)
        )
        self.board_frame_label.grid(
            row=3, column=0, padx=10, pady=10, sticky="w"
        )

        self.board_frame_text = tk.Text(
            frame, height=3, width=30, font=("Helvetica", 10)
        )
        self.board_frame_text.grid(
            row=4, column=0, padx=10, pady=10, sticky="ew"
        )
        # self.board_frame_text.config(state='disabled')

        # Create a copy button for the board frame text
        self.copy_board_frame_button = ttk.Button(
            frame,
            text="Copy",
            command=lambda: self.copy_to_clipboard(
                self.board_frame_text.get("1.0", "end-1c")
            ),
        )
        self.copy_board_frame_button.grid(row=4, column=1, padx=10, pady=10)

        # create a goto button
        self.goto_board_frame_button = ttk.Button(
            frame,
            text="Goto",
            command=lambda: self.goto_board_frame_cb(
                self.board_frame_text.get("1.0", "end-1c")
            ),
        )
        self.goto_board_frame_button.grid(row=4, column=2, padx=10, pady=10)

        # Create a text box to display the current tool tip pose in joint angles
        self.joint_angles_label = tk.Label(
            frame, text="Joint Angles", font=("Helvetica", 10)
        )
        self.joint_angles_label.grid(
            row=5, column=0, padx=10, pady=10, sticky="w"
        )

        self.joint_angles_text = tk.Text(
            frame, height=3, width=30, font=("Helvetica", 10)
        )
        self.joint_angles_text.grid(
            row=6, column=0, padx=10, pady=10, sticky="ew"
        )
        # self.joint_angles_text.config(state='disabled')

        # Create a copy button for the joint angles text
        self.copy_joint_angles_button = ttk.Button(
            frame,
            text="Copy",
            command=lambda: self.copy_to_clipboard(
                self.joint_angles_text.get("1.0", "end-1c")
            ),
        )
        self.copy_joint_angles_button.grid(row=6, column=1, padx=10, pady=10)

        # create a goto button
        self.goto_joint_angles_button = ttk.Button(
            frame,
            text="Goto",
            command=lambda: self.goto_joint_angles_cb(
                self.joint_angles_text.get("1.0", "end-1c")
            ),
        )
        self.goto_joint_angles_button.grid(row=6, column=2, padx=10, pady=10)

        # Create an update button
        self.update_button = ttk.Button(
            frame, text="Update", command=self.update_tool_tip_pose
        )
        self.update_button.grid(row=7, column=0, padx=10, pady=10, columnspan=2)

        # Set button style
        style = ttk.Style()
        style.configure("TButton", font=("Helvetica", 12))
        style.map(
            "TButton",
            background=[("active", "#0056b3"), ("selected", "#0056b3")],
            foreground=[("active", "#ffffff"), ("selected", "#ffffff")],
        )

        frame.grid_columnconfigure(0, weight=1)

    def copy_to_clipboard(self, text):
        # Function to copy text to the clipboard
        self.master.clipboard_clear()
        self.master.clipboard_append(text)

    def goto_base_frame_cb(self, text):
        # parse the text and send the robot to the pose
        # text format: "[x, y, z, theta_x, theta_y, theta_z]"
        # get the values from the text
        values = text[1:-1].split(",")
        values = [float(i) for i in values]

        # create a pose
        kp = get_kinovapose_from_list(values)

        # send the robot to the pose
        self.arm.clear_faults()
        rospy.sleep(0.5)
        if not self.arm.send_cartesian_pose(kp):
            rospy.logerr("Failed to send pose")

        messagebox.showinfo("Base Frame", "Pose sent to robot")

    def goto_board_frame_cb(self, text):
        # text format: "[x, y, z, qx, qy, qz, qw]"
        # get the values from the text
        values = text[1:-1].split(",")
        values = [float(i) for i in values]

        # create a pose
        msg = PoseStamped()
        msg.header.frame_id = "board_link"
        msg.pose.position.x = values[0]
        msg.pose.position.y = values[1]
        msg.pose.position.z = values[2]
        msg.pose.orientation.x = values[3]
        msg.pose.orientation.y = values[4]
        msg.pose.orientation.z = values[5]
        msg.pose.orientation.w = values[6]

        msg = self.transform_utils.transformed_pose_with_retries(
            msg, "base_link"
        )

        # send the robot to the pose
        self.arm.clear_faults()
        rospy.sleep(0.5)
        kp = get_kinovapose_from_pose_stamped(msg)
        if not self.arm.send_cartesian_pose(kp):
            rospy.logerr("Failed to send pose")

        messagebox.showinfo("Board Frame", "Pose sent to robot")

    def goto_joint_angles_cb(self, text):
        # text format: "[j1, j2, j3, j4, j5, j6, j7, j8]"
        # get the values from the text
        values = text[1:-1].split(",")
        # get only the first 7 values
        values = [float(i) for i in values[:7]]

        # send the robot to the pose
        self.arm.clear_faults()
        rospy.sleep(0.5)
        if not self.arm.send_joint_angles(values):
            rospy.logerr("Failed to send joint angles")

        messagebox.showinfo("Joint Angles", "Joint angles sent to robot")

    def update_tool_tip_pose(self):
        # Function to update the current tool tip pose in three formats
        # Get the current tool tip pose
        current_pose = self.arm.get_current_pose()
        cp_list = current_pose.to_list()

        rospy.loginfo("got current pose")

        cp_bl = current_pose.to_pose_stamped()
        cp_in_board_frame = self.transform_utils.transformed_pose_with_retries(
            cp_bl, "board_link"
        )

        if cp_in_board_frame is not None:
            cp_in_bf_pos = cp_in_board_frame.pose.position
            cp_in_bf_ori = cp_in_board_frame.pose.orientation

            cp_in_bf = [
                cp_in_bf_pos.x,
                cp_in_bf_pos.y,
                cp_in_bf_pos.z,
                cp_in_bf_ori.x,
                cp_in_bf_ori.y,
                cp_in_bf_ori.z,
                cp_in_bf_ori.w,
            ]

            rospy.loginfo("got current pose in board frame")
        else:
            cp_in_bf = "Could not get pose in board frame, check tf tree"

        # get joint angles from joint state publisher
        # TODO: change the topic
        msg: JointState = rospy.wait_for_message(
            "/my_gen3/base_feedback/joint_state", JointState
        )
        joint_angles = [math.degrees(angle) for angle in msg.position]

        rospy.loginfo("got joint angles")

        # Update the text boxes with the current pose
        self.base_frame_text.config(state="normal")
        self.base_frame_text.delete("1.0", tk.END)
        self.base_frame_text.insert(tk.END, str(cp_list))
        # self.base_frame_text.config(state='disabled')

        self.board_frame_text.config(state="normal")
        self.board_frame_text.delete("1.0", tk.END)
        self.board_frame_text.insert(tk.END, str(cp_in_bf))
        # self.board_frame_text.config(state='disabled')

        self.joint_angles_text.config(state="normal")
        self.joint_angles_text.delete("1.0", tk.END)
        self.joint_angles_text.insert(tk.END, str(joint_angles))
        # self.joint_angles_text.config(state='disabled')

    def create_window(self):
        # set the notebook
        self.noetbook = ttk.Notebook(self.master)
        self.noetbook.grid(
            row=0, column=0, padx=10, pady=10, sticky="nsew"
        )  # Updated to use grid manager
        self.master.grid_rowconfigure(
            0, weight=1
        )  # Configure row 0 to take all extra space

        # Create and configure the window's contents
        self.render_lists(self.noetbook)

        gripper_frame = tk.Frame(self.master)
        gripper_frame.grid(
            row=1, column=0, padx=10, pady=10, sticky="nsew"
        )  # Updated to use grid manager
        self.master.grid_rowconfigure(
            1, weight=0
        )  # Configure row 1 to take no height

        self.gripper_command_window(gripper_frame)

        pose_frame = tk.Frame(self.master)
        pose_frame.grid(
            row=2, column=0, padx=10, pady=10, sticky="nsew"
        )  # Updated to use grid manager
        self.master.grid_rowconfigure(
            2, weight=0
        )  # Configure row 2 to take no height

        # Call the current_pose_window method with the pose_frame as the parent
        self.current_pose_window(pose_frame)

        # Configure the columns to take 100% width
        self.master.grid_columnconfigure(0, weight=1)

        t = threading.Thread(target=self.master.mainloop())
        t.daemon = True
        t.start()

    def on_shutdown(self):
        rospy.loginfo("Shutting down")
        self.master.destroy()


if __name__ == "__main__":
    rospy.init_node("gui_kinova_node")
    task = GuiKinova()
    rospy.on_shutdown(task.on_shutdown)
    try:
        task.create_window()
        rospy.spin()
    except rospy.ROSInterruptException or KeyboardInterrupt:
        rospy.logerr(
            "Got ROSInterruptException or KeyboardInterrupt, shutting down"
        )
        rospy.signal_shutdown("GUI interrupted")
        sys.exit(0)
