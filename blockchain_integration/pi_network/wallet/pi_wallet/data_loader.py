import pandas as pd
import yfinance as yf


class DataLoader:
    def __init__(self, ticker, start_date, end_date):
        self.ticker = ticker
        self.start_date = start_date
        self.end_date = end_date

    def load_data(self):
        data = yf.download(self.ticker, start=self.start_date, end=self.end_date)
        data.reset_index(inplace=True)
        data.rename(
            columns={
                "Date": "date",
                "Open": "open",
                "High": "high",
                "Low": "low",
                "Close": "close",
                "Adj Close": "adj_close",
                "Volume": "volume",
            },
            inplace=True,
        )
        return data

    def preprocess_data(self, data):
        data["return"] = data["close"].pct_change()
        data["log_return"] = np.log(data["close"] / data["close"].shift(1))
        data.dropna(inplace=True)
        return data

    def split_data(self, data):
        train_data, test_data = data.split(test_size=0.2, random_state=42)
        return train_data, test_data
