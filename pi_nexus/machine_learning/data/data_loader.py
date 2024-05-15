# machine_learning/data/data_loader.py
import pandas as pd
from .datasets import Datasets

class DataLoader:
    def __init__(self):
        self.datasets = Datasets()

    def load_fraud_data(self):
        return self.datasets.fraud_data

    def load_user_behavior_data(self):
        return self.datasets.user_behavior_data

    def load_system_performance_data(self):
        return self.datasets.system_performance_data
