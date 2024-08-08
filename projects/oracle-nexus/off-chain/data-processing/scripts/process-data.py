import os
import json
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
from oracle_nexus import OracleNexus

# Load configuration
with open('config.json') as f:
    config = json.load(f)

# Set up Oracle Nexus contract
oracle_nexus = OracleNexus(config['oracle_nexus_address'], config['oracle_nexus_abi'])

# Function to load data from Oracle Nexus contract
def load_data():
    data = oracle_nexus.getData()
    return pd.DataFrame(json.loads(data))

# Function to preprocess data
def preprocess_data(df):
    # Handle missing values
    df.fillna(df.mean(), inplace=True)

    # Scale data
    scaler = StandardScaler()
    df[['feature1', 'feature2', 'feature3']] = scaler.fit_transform(df[['feature1', 'feature2', 'feature3']])

    # Oversample minority class
    smote = SMOTE(random_state=42)
    df, _ = smote.fit_resample(df, df['target'])

    return df

# Function to train model
def train_model(df):
    X = df.drop('target', axis=1)
    y = df['target']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    print('Model accuracy:', accuracy_score(y_test, y_pred))
    print('Model classification report:')
    print(classification_report(y_test, y_pred))

    return model

# Function to deploy model to Oracle Nexus contract
def deploy_model(model):
    model_bytes = pickle.dumps(model)
    oracle_nexus.deployModel(model_bytes)

# Main script
if __name__ == '__main__':
    # Load data from Oracle Nexus contract
    df = load_data()

    # Preprocess data
    df = preprocess_data(df)

    # Train model
    model = train_model(df)

    # Deploy model to Oracle Nexus contract
    deploy_model(model)
