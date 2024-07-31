import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler, LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

def preprocess_data(data, categorical_cols, numerical_cols, text_cols, target_col):
    """
    Preprocess the data by handling missing values, encoding categorical variables, 
    scaling numerical variables, and transforming text variables.

    Args:
        data (pd.DataFrame): The input data
        categorical_cols (list): List of categorical column names
        numerical_cols (list): List of numerical column names
        text_cols (list): List of text column names
        target_col (str): Name of the target column

    Returns:
        pd.DataFrame: Preprocessed data
    """
    # Handle missing values
    imputer = SimpleImputer(strategy='median')
    data[numerical_cols] = imputer.fit_transform(data[numerical_cols])

    # Encode categorical variables
    encoder = LabelEncoder()
    for col in categorical_cols:
        data[col] = encoder.fit_transform(data[col])

    # Scale numerical variables
    scaler = StandardScaler()
    data[numerical_cols] = scaler.fit_transform(data[numerical_cols])

    # Transform text variables
    vectorizer = TfidfVectorizer(max_features=5000)
    for col in text_cols:
        data[col] = vectorizer.fit_transform(data[col])

    # Apply PCA to reduce dimensionality
    pca = PCA(n_components=0.95)
    data[numerical_cols] = pca.fit_transform(data[numerical_cols])

    # Create a ColumnTransformer to combine preprocessing steps
    preprocessing_steps = [
        ('imputer', imputer, numerical_cols),
        ('encoder', encoder, categorical_cols),
        ('scaler', scaler, numerical_cols),
        ('vectorizer', vectorizer, text_cols),
        ('pca', pca, numerical_cols)
    ]
    preprocessor = ColumnTransformer(preprocessing_steps)

    # Apply the preprocessing pipeline
    data_preprocessed = preprocessor.fit_transform(data)

    # Convert the preprocessed data back to a DataFrame
    data_preprocessed = pd.DataFrame(data_preprocessed, columns=data.columns)

    return data_preprocessed

def split_data(data, target_col, test_size=0.2, random_state=42):
    """
    Split the data into training and testing sets.

    Args:
        data (pd.DataFrame): The input data
        target_col (str): Name of the target column
        test_size (float): Proportion of data to use for testing
        random_state (int): Random seed for reproducibility

    Returns:
        X_train, X_test, y_train, y_test (pd.DataFrame, pd.DataFrame, pd.Series, pd.Series): 
            Training and testing data
    """
    X = data.drop(target_col, axis=1)
    y = data[target_col]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
    return X_train, X_test, y_train, y_test
