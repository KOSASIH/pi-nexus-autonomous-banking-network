import pandas as pd
import matplotlib.pyplot as plt

class NexusDataAnalyticsDashboard:
    def __init__(self):
        self.data = pd.read_csv('data.csv')

    def display_dashboard(self):
        plt.plot(self.data['column1'], self.data['column2'])
        plt.show()

if __name__ == '__main__':
    dashboard = NexusDataAnalyticsDashboard()
    dashboard.display_dashboard()
