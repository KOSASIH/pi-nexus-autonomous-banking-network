# analytics_dashboard.py
import matplotlib.pyplot as plt
import pandas as pd


class AnalyticsDashboard:
    def __init__(self, data_source):
        self.data_source = data_source

    def update_dashboard(self):
        data = self.data_source.get_data()
        df = pd.DataFrame(data)
        plt.plot(df["timestamp"], df["transaction_volume"])
        plt.xlabel("Time")
        plt.ylabel("Transaction Volume")
        plt.title("Real-time Transaction Volume")
        plt.show()
