# data_preprocessor.py

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from typing import Any, Dict, List, Optional

class DataPreprocessor:
    """Data preprocessor class."""

    def __init__(self, numerical_features: List[str], categorical_features: List[str]):
        self.numerical_features = numerical_features
        self.categorical_features = categorical_features
        self.preprocessor = self._create_preprocessor()

    def _create_preprocessor(self) -> Pipeline:
        """Create a preprocessing pipeline."""
        numerical_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', StandardScaler()),
        ])
        categorical_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
            ('encoder', LabelEncoder()),
        ])
        preprocessor = ColumnTransformer(
            transformers=[
                ('numerical', numerical_transformer, self.numerical_features),
                ('categorical', categorical_transformer, self.categorical_features),
            ]
        )
        return preprocessor

    def preprocess_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """Preprocess the data."""
        return pd.DataFrame(self.preprocessor.fit_transform(data), columns=data.columns)

def main():
    data = pd.DataFrame({
        'feature1': [1, 2, 3, 4, 5],
        'feature2': [10, 20, 30, 40, 50],
        'feature3': ['a', 'b', 'c', 'd', 'e'],
    })
    preprocessor = DataPreprocessor(numerical_features=['feature1', 'feature2'], categorical_features=['feature3'])
    preprocessed_data = preprocessor.preprocess_data(data)
    print(preprocessed_data.head())

if __name__ == '__main__':
    main()
