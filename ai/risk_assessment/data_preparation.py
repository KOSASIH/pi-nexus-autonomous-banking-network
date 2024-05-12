import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


def load_data(data_path):
    """
    Loads the data from the given file path.
    """
    data = pd.read_csv(data_path)
    return data


def preprocess_data(data):
    """
    Preprocesses the data by cleaning, transforming, and encoding the features.
    """
    # Clean the data
    data = data.dropna()

    # Transform the data
    data["amount"] = np.log(data["amount"])

    # Encode the categorical features
    data = pd.get_dummies(data, columns=["category", "merchant"])

    return data


def split_data(data, test_size=0.2):
    """
    Splits the data into training and testing sets.
    """
    X = data.drop("fraud", axis=1)
    y = data["fraud"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=42
    )
    return X_train, X_test, y_train, y_test
