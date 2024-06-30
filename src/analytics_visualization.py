import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

class AnalyticsVisualization:
    def __init__(self, data):
        self.data = data

    def scatter_plot(self, x, y, c=None, s=None):
        plt.scatter(self.data[x], self.data[y], c=self.data[c], s=self.data[s])
        plt.title("Scatter Plot")
        plt.xlabel(x)
        plt.ylabel(y)
        plt.colorbar()
        plt.show()

    def line_chart(self, x, y):
        sns.lineplot(x=x, y=y, data=self.data)
        plt.title("Line Chart")
        plt.xlabel(x)
        plt.ylabel(y)
        plt.show()

    def seaborn_scatter_plot(self, x, y, hue=None):
        sns.scatterplot(x=x, y=y, data=self.data, hue=hue)
        plt.title("Seaborn Scatter Plot")
        plt.xlabel(x)
        plt.ylabel(y)
        plt.show()

# Load the tips dataset
data = pd.read_csv("tips.csv")

# Create an instance of the AnalyticsVisualization class
analytics_visualization = AnalyticsVisualization(data)

# Call the scatter plot method
analytics_visualization.scatter_plot('day', 'tip', c='size', s='total_bill')

# Call the line chart method
analytics_visualization.line_chart('sex', 'total_bill')

# Call the seaborn scatter plot method
analytics_visualization.seaborn_scatter_plot('day', 'tip', hue='sex')
