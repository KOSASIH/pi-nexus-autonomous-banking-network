import pandas as pd


class DataIngestion:
    def __init__(self, data_source):
        self.data_source = data_source

    def ingest_data(self):
        data = pd.read_csv(self.data_source)
        return data
