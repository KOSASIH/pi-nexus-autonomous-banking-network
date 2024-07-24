import numpy as np
import tensorflow as tf
from tensorflow import keras
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import os
import random

class EnvironmentGenerator:
    def __init__(self, width=100, height=100, depth=100):
        self.width = width
        self.height = height
        self.depth = depth
        self.environment = np.zeros((width, height, depth))
        self.scaler = MinMaxScaler()

    def generate_terrain(self):
        x = np.linspace(0, self.width, self.width)
        y = np.linspace(0, self.height, self.height)
        x, y = np.meshgrid(x, y)
        z = np.sin(x) * np.cos(y) + np.random.normal(0, 0.1, (self.width, self.height))
        self.environment[:, :, 0] = z

    def generate_vegetation(self):
        for i in range(self.width):
            for j in range(self.height):
                if self.environment[i, j, 0] > 0.5:
                    self.environment[i, j, 1] = 1
                else:
                    self.environment[i, j, 1] = 0

    def generate_water(self):
        for i in range(self.width):
            for j in range(self.height):
                if self.environment[i, j, 0] < 0.2:
                    self.environment[i, j, 2] = 1
                else:
                    self.environment[i, j, 2] = 0

    def generate_roads(self):
        for i in range(self.width):
            for j in range(self.height):
                if i % 10 == 0 or j % 10 == 0:
                    self.environment[i, j, 3] = 1
                else:
                    self.environment[i, j, 3] = 0

    def generate_buildings(self):
        for i in range(self.width):
            for j in range(self.height):
                if self.environment[i, j, 1] == 1 and self.environment[i, j, 3] == 0:
                    if random.random() < 0.1:
                        self.environment[i, j, 4] = 1
                    else:
                        self.environment[i, j, 4] = 0

    def generate_environment(self):
        self.generate_terrain()
        self.generate_vegetation()
        self.generate_water()
        self.generate_roads()
        self.generate_buildings()

    def visualize_environment(self):
        fig = plt.figure(figsize=(10, 10))
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(self.environment[:, :, 0].flatten(), self.environment[:, :, 1].flatten(), self.environment[:, :, 2].flatten(), c=self.environment[:, :, 3].flatten())
        plt.show()

    def save_environment(self, filename):
        np.save(filename, self.environment)

    def load_environment(self, filename):
        self.environment = np.load(filename)

if __name__ == '__main__':
    generator = EnvironmentGenerator()
    generator.generate_environment()
    generator.visualize_environment()
    generator.save_environment('environment.npy')
