# Importing necessary libraries
import rospy
from geometry_msgs.msg import Twist

# Class for robotics
class PiNetworkRobotics:
    def __init__(self):
        self.pub = rospy.Publisher('cmd_vel', Twist, queue_size=10)
        rospy.init_node('pi_network_robotics')

    # Function to move the robot
    def move_robot(self, linear_velocity, angular_velocity):
        twist = Twist()
        twist.linear.x = linear_velocity
        twist.angular.z = angular_velocity
        self.pub.publish(twist)

# Example usage
robot = PiNetworkRobotics()
robot.move_robot(0.5, 0.2)
