import pandas as pd

def process_data(data):
    # Process the data using pandas
    df = pd.DataFrame(data)
    df = df.dropna()
    return df

def load_data(file_path):
    # Load the data from a CSV file
    df = pd.read_csv(file_path)
    return df
