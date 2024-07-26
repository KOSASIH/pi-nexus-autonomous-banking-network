# sidra_chain_robotics.py
import rospy
from gazebo_msgs.msg import ModelState
from gazebo_msgs.srv import SetModelState
import numpy as np
from scipy.spatial.transform import Rotation as R

class SidraChainRobotics:
    def __init__(self):
        # Initialize ROS node
        rospy.init_node("sidra_chain_robotics")

        # Create ROS publishers and services
        self.model_state_pub = rospy.Publisher("gazebo/set_model_state", ModelState, queue_size=10)
        self.set_model_state_srv = rospy.ServiceProxy("gazebo/set_model_state", SetModelState)

        # Initialize robot state
        self.robot_state = {"x": 0.0, "y": 0.0, "z": 0.0, "roll": 0.0, "pitch": 0.0, "yaw": 0.0}

    def simulate_robot(self):
        # Simulate robot movement
        while not rospy.is_shutdown():
            # Update robot state
            self.update_robot_state()

            # Create ModelState message
            model_state = ModelState()
            model_state.model_name = "sidra_chain_robot"
            model_state.pose.position.x = self.robot_state["x"]
            model_state.pose.position.y = self.robot_state["y"]
            model_state.pose.position.z = self.robot_state["z"]
            model_state.pose.orientation.x = self.robot_state["roll"]
            model_state.pose.orientation.y = self.robot_state["pitch"]
            model_state.pose.orientation.z = self.robot_state["yaw"]

            # Publish ModelState message
            self.model_state_pub.publish(model_state)

            # Call SetModelState service
            self.set_model_state_srv(model_state)

            # Sleep for simulation time step
            rospy.sleep(0.01)
