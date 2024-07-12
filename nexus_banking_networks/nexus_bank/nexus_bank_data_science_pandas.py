import pandas as pd

class DataScience:
    def __init__(self):
        self.df = pd.DataFrame({
            'Name': ['Alice', 'Bob', 'Charlie'],
            'Age': [25, 30, 35],
            'Country': ['USA', 'Canada', 'Mexico']
        })

    def analyze_data(self):
        print(self.df.describe())
        print(self.df.groupby('Country').mean())
        print(self.df.corr())

ds = DataScience()
ds.analyze_data()
