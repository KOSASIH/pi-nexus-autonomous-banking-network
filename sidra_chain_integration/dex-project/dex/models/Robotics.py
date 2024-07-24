import rospy
from geometry_msgs.msg import Twist

class Robotics:
    def __init__(self, robot_name):
        self.robot_name = robot_name
        self.publisher = rospy.Publisher(f'/{robot_name}/cmd_vel', Twist, queue_size=10)

    def move_forward(self, speed):
        twist = Twist()
        twist.linear.x = speed
        self.publisher.publish(twist)

    def turn_left(self, speed):
        twist = Twist()
        twist.angular.z = speed
        self.publisher.publish(twist)

    def stop(self):
        twist = Twist()
        self.publisher.publish(twist)
