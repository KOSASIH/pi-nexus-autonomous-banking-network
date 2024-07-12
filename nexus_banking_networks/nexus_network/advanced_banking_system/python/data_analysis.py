# data_analysis.py
import pandas as pd
import matplotlib.pyplot as plt

class DataAnalysis:
    def __init__(self):
        self.data = pd.read_csv('banking_data.csv')

    def visualize_data(self):
        self.data.plot(kind='bar')
        plt.show()

    def calculate_statistics(self):
        return self.data.describe()
