import pandas as pd
import numpy as np
import datetime as dt
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

class TransactionAnalysis:
    def __init__(self, transaction_data: pd.DataFrame):
        self.transaction_data = transaction_data

    def preprocess_data(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        # Preprocess the data by cleaning, transforming, and splitting it into features and labels
        self.transaction_data['timestamp'] = pd.to_datetime(self.transaction_data['timestamp'])
        self.transaction_data['hour'] = self.transaction_data['timestamp'].dt.hour
        self.transaction_data['day'] = self.transaction_data['timestamp'].dt.day
        self.transaction_data['month'] = self.transaction_data['timestamp'].dt.month
        self.transaction_data['year'] = self.transaction_data['timestamp'].dt.year

        categorical_features = ['card_type', 'country', 'erchant_category']
        numerical_features = ['amount', 'hour', 'day', 'onth', 'year']

        categorical_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='most_frequent')),
            ('onehot', pd.get_dummies)
        ])

        numerical_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='mean')),
            ('scaler', StandardScaler())
        ])

        preprocessor = ColumnTransformer(
            transformers=[
                ('num', numerical_transformer, numerical_features),
                ('cat', categorical_transformer, categorical_features)
            ]
        )

        self.transaction_data = preprocessor.fit_transform(self.transaction_data)

        X = self.transaction_data.drop('is_fraud', axis=1)
        y = self.transaction_data['is_fraud']

        return X, y

    def train_model(self, X: pd.DataFrame, y: pd.Series) -> xgb.XGBClassifier:
        # Train an XGBoost model using the features and labels
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        model = xgb.XGBClassifier(objective='binary:logistic', max_depth=6, learning_rate=0.1, n_estimators=1000, n_jobs=-1)
        model.fit(X_train, y_train)

        return model

    def evaluate_model(self, model: xgb.XGBClassifier, X: pd.DataFrame, y: pd.Series) -> Tuple[float, float, float, float]:
        # Evaluate the model using accuracy, precision, recall, and F1 score
        y_pred = model.predict(X)
        accuracy = accuracy_score(y, y_pred)
        precision = precision_score(y, y_pred)
        recall = recall_score(y, y_pred)
        f1 = f1_score(y, y_pred)

        return accuracy, precision, recall, f1

    def analyze_transactions(self, model: xgb.XGBClassifier, transactions: pd.DataFrame) -> pd.DataFrame:
        # Analyze the transactions using the trained model and return a dataframe with the predictions
        transactions = transactions.drop('is_fraud', axis=1)
        predictions = model.predict(transactions)

        return pd.DataFrame({'transaction_id': transactions.index, 'is_fraud': predictions})

    def visualize_results(self, transactions: pd.DataFrame) -> None:
        # Visualize the results using a heatmap and a bar chart
        import seaborn as sns
        import matplotlib.pyplot as plt

        plt.figure(figsize=(10, 8))
        sns.heatmap(transactions.corr(), annot=True, cmap='coolwarm', square=True)
        plt.title('Correlation Matrix')
        plt.show()

        plt.figure(figsize=(10, 8))
        sns.countplot(x='is_fraud', data=transactions)
        plt.title('Fraud Distribution')
        plt.show()

if __name__ == "__main__":
    # Load the transaction data
    transaction_data = pd.read_csv('transactions.csv')

    # Create an instance of the TransactionAnalysis class
    analysis = TransactionAnalysis(transaction_data)

    # Preprocess the data
    X, y = analysis.preprocess_data()

    # Train the model
    model = analysis.train_model(X, y)

    # Evaluate the model
    accuracy, precision, recall, f1 = analysis.evaluate_model(model, X, y)
    print(f'Accuracy: {accuracy:.3f}, Precision: {precision:.3f}, Recall: {recall:.3f}, F1: {f1:.3f}')

    # Analyze new transactions
    new_transactions = pd.read_csv('new_transactions.csv')
    results = analysis.analyze_transactions(model, new_transactions)

    # Visualize the results
    analysis.visualize_results(results)
