# Importing necessary libraries
import rospy
from geometry_msgs.msg import Twist

# Class for robotics
class PiNetworkRobotics2:
    def __init__(self):
        self.pub = rospy.Publisher('cmd_vel', Twist, queue_size=10)
        rospy.init_node('pi_network_robotics2')

    # Function to move the robot
    def move_robot(self, linear_velocity, angular_velocity):
        twist = Twist()
        twist.linear.x = linear_velocity
        twist.angular.z = angular_velocity
        self.pub.publish(twist)

    # Function to stop the robot
    def stop_robot(self):
        twist = Twist()
        twist.linear.x = 0
        twist.angular.z = 0
        self.pub.publish(twist)

# Example usage
robot2 = PiNetworkRobotics2()
robot2.move_robot(0.5, 0.2)
robot2.stop_robot()
