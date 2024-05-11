import pandas as pd

class DataPreparation:
    def __init__(self, data_file):
        self.data_file = data_file

    def prepare_data(self):
        """
        Prepares the data for user behavior analysis.
        """
        data = pd.read_csv(self.data_file)
        data = data.dropna()
        return data
