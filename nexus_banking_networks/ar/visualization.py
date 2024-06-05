import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


class Visualization:

    def __init__(self, data):
        self.data = data

    def visualize_data(self):
        # Visualize data using AR
        fig = plt.figure()
        ax = fig.add_subplot(111, projection="3d")
        ax.scatter(self.data["x"], self.data["y"], self.data["z"])
        plt.show()
