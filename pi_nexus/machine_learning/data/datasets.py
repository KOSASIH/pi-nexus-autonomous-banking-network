# machine_learning/data/datasets.py
import pandas as pd


class Datasets:
    def __init__(self):
        self.fraud_data = pd.read_csv("fraud_data.csv")
        self.user_behavior_data = pd.read_csv("user_behavior_data.csv")
        self.system_performance_data = pd.read_csv("system_performance_data.csv")
