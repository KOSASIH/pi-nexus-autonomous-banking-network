import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

class DataVisualization:
    def __init__(self, data):
        self.data = data

    def visualize_data(self):
        plt.figure(figsize=(10, 6))
        plt.plot(self.data["Date"], self.data["Value"])
        plt.xlabel("Date")
        plt.ylabel("Value")
        plt.title("Data Visualization")
        plt.show()

# Example usage:
data = pd.read_csv("data.csv")
data_visualization = DataVisualization(data)
data_visualization.visualize_data()
