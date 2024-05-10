import numpy as np
import pandas as pd


def preprocess_data(data):
    """
    Preprocesses the specified data by cleaning, transforming, and scaling it.
    """
    # Clean missing values
    data = data.dropna()

    # Transform categorical variables to numerical variables
    data = pd.get_dummies(data)

    # Scale numerical variables to have zero mean and unit variance
    scaler = StandardScaler()
    data = pd.DataFrame(scaler.fit_transform(data), columns=data.columns)

    return data


def split_data(data, test_size=0.2):
    """
    Splits the specified data into training and testing sets.
    """
    np.random.seed(42)
    shuffled_index = np.random.permutation(len(data))
    test_set_size = int(len(data) * test_size)
    test_indices = shuffled_index[:test_set_size]
    train_indices = shuffled_index[test_set_size:]

    train_data = data.iloc[train_indices]
    test_data = data.iloc[test_indices]

    return train_data, test_data
