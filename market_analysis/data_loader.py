import pandas as pd
import yfinance as yf

class DataLoader:
    def __init__(self, ticker, start_date, end_date):
        self.ticker = ticker
        self.start_date = start_date
        self.end_date = end_date
        self.data = self.load_data()

    def load_data(self):
        data = yf.download(self.ticker, start=self.start_date, end=self.end_date)
        data.reset_index(inplace=True)
        data.rename(columns={'Date': 'date', 'Open': 'open', 'High': 'high', 'Low': 'low', 'Close': 'close', 'Volume': 'volume'}, inplace=True)
        return data

    def preprocess_data(self):
        data = self.data.copy()
        data['date'] = pd.to_datetime(data['date'])
        data.set_index('date', inplace=True)
        data.dropna(inplace=True)
        return data
