# visualization/enhanced_visualization.py
import matplotlib.pyplot as plt

class EnhancedVisualization:
    def __init__(self, data):
        self.data = data

    def visualize_data(self):
        # Use Matplotlib to create interactive visualizations
        plt.plot(self.data)
        plt.show()
