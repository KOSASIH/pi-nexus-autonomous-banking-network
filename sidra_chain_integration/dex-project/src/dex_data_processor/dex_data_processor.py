# dex_data_processor.py
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

class DexDataProcessor:
    def __init__(self):
        self.scaler = MinMaxScaler()

    def process_dex_data(self, data):
        # Process DEX data
        df = pd.DataFrame(data)
        df['value'] = self.scaler.fit_transform(df['value'].values.reshape(-1, 1))
        return df.to_dict('records')

    def analyze_dex_data(self, data):
        # Analyze DEX data
        df = pd.DataFrame(data)
        analysis = df.describe()
        return analysis.to_dict()
