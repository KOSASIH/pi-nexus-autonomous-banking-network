# visualization.py

import os
import json
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
from plotly.offline import iplot
from sklearn.metrics import confusion_matrix

class Visualization:
    def __init__(self, data_file):
        self.data_file = data_file
        self.data = self.load_data()

    def load_data(self):
        with open(self.data_file, 'r') as f:
            data = json.load(f)
        return data

    def plot_confusion_matrix(self, y_true, y_pred, title='Confusion Matrix'):
        cm = confusion_matrix(y_true, y_pred)
        plt.imshow(cm, interpolation='nearest')
        plt.title(title)
        plt.colorbar()
        plt.xlabel('Predicted labels')
        plt.ylabel('True labels')
        plt.show()

    def plot_bar_chart(self, data, x_axis, y_axis, title='Bar Chart'):
        sns.barplot(x=x_axis, y=y_axis, data=data)
        plt.title(title)
        plt.show()

    def plot_line_chart(self, data, x_axis, y_axis, title='Line Chart'):
        sns.lineplot(x=x_axis, y=y_axis, data=data)
        plt.title(title)
        plt.show()

    def plot_scatter_plot(self, data, x_axis, y_axis, title='Scatter Plot'):
        sns.scatterplot(x=x_axis, y=y_axis, data=data)
        plt.title(title)
        plt.show()

    def plot_3d_scatter_plot(self, data, x_axis, y_axis, z_axis, title='3D Scatter Plot'):
        fig = go.Figure(data=[go.Scatter3d(
            x=data[x_axis],
            y=data[y_axis],
            z=data[z_axis],
            mode='markers',
            marker=dict(
                size=5,
                color=data[z_axis],
                colorscale='Viridis',
                showscale=True
            )
        )])
        fig.update_layout(title=title, scene=dict(
            xaxis_title=x_axis,
            yaxis_title=y_axis,
            zaxis_title=z_axis
        ))
        iplot(fig)

visualization = Visualization('github_api_data.json')
