import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

class Robot:
    def __init__(self, x, y, z, theta):
        self.x = x
        self.y = y
        self.z = z
        self.theta = theta

    def move_forward(self, distance):
        self.x += distance * np.cos(self.theta)
        self.y += distance * np.sin(self.theta)

    def turn_left(self, angle):
        self.theta += angle

    def turn_right(self, angle):
        self.theta -= angle

    def get_position(self):
        return self.x, self.y, self.z

    def plot_path(self):
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(self.x, self.y, self.z)
        plt.show()

robot = Robot(0, 0, 0, 0)
robot.move_forward(10)
robot.turn_left(np.pi/2)
robot.move_forward(5)
robot.turn_right(np.pi/4)
robot.move_forward(3)
robot.plot_path()
