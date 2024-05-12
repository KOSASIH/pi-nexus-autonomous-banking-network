import pandas as pd


class DataPreparation:
    def __init__(self, data_path):
        self.data_path = data_path

    def prepare_data(self):
        """
        Prepares the data for training the machine learning model.
        """
        data = pd.read_csv(self.data_path)
        prepared_data = self.preprocess_data(data)
        return prepared_data

    def preprocess_data(self, data):
        """
        Preprocesses the data by cleaning, transforming, and normalizing it.
        """
        # Preprocessing steps
        data = data.dropna()
        data = data.drop_duplicates()
        data = (data - data.min()) / (data.max() - data.min())
        return data
