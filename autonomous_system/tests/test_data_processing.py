import pandas as pd
import pytest
from data_cleaning import clean_data
from data_transformation import transform_data


def test_clean_data():
    # Load sample data
    input_data = pd.read_csv("input_data.csv")

    # Clean data
    cleaned_data = clean_data(input_data)

    # Check that the data has been cleaned correctly
    assert cleaned_data.isnull().sum().sum() == 0
    assert cleaned_data.duplicated().sum() == 0


def test_transform_data():
    # Load sample data
    input_data = pd.read_csv("input_data.csv")

    # Clean data
    cleaned_data = clean_data(input_data)

    # Transform data
    transformed_data = transform_data(cleaned_data)

    # Check that the data has been transformed correctly
    assert transformed_data.shape[1] == 10
    assert transformed_data.isnull().sum().sum() == 0
