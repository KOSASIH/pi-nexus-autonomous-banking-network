import pandas as pd
from sklearn.preprocessing import StandardScaler

from config import get_config


def transform_data():
    # Load cleaned data from CSV file
    input_file = get_config("data_processing.input_file")
    df = pd.read_csv(input_file)

    # Extract features and target variables
    X = df[["feature1", "feature2", "feature3"]]
    y = df["target"]

    # Scale the features using StandardScaler
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Save transformed data to CSV file
    output_file = get_config("data_processing.output_file")
    transformed_df = pd.DataFrame(
        X_scaled, columns=["feature1", "feature2", "feature3"]
    )
    transformed_df["target"] = y
    transformed_df.to_csv(output_file, index=False)


if __name__ == "__main__":
    transform_data()
