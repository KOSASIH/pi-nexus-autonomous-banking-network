import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

from config import get_config


def train_model():
    # Load transformed data from CSV file
    input_file = get_config("data_processing.input_file")
    df = pd.read_csv(input_file)

    # Extract features and target variables
    X = df[["feature1", "feature2", "feature3"]]
    y = df["target"]

    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Train the model using Logistic Regression
    model = LogisticRegression()
    model.fit(X_train, y_train)

    # Save the trained model to a pickle file
    output_file = get_config("model_training.output_file")
    model.save(output_file)


if __name__ == "__main__":
    train_model()
