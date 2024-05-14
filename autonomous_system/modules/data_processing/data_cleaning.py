import pandas as pd
from config import get_config

def clean_data():
    # Load input data from CSV file
    input_file = get_config('data_processing.input_file')
    df = pd.read_csv(input_file)

    # Drop any rows with missing values
    df.dropna(inplace=True)

    # Save cleaned data to CSV file
    output_file = get_config('data_processing.output_file')
    df.to_csv(output_file, index=False)

if __name__ == '__main__':
    clean_data()
