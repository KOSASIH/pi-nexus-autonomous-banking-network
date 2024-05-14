import pytest
from model_evaluation import evaluate_model
from model_training import train_model
from sklearn.model_selection import train_test_split


def test_train_model():
    # Load sample data
    input_data = pd.read_csv("input_data.csv")

    # Clean data
    cleaned_data = clean_data(input_data)

    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(
        cleaned_data.drop("target", axis=1),
        cleaned_data["target"],
        test_size=0.2,
        random_state=42,
    )

    # Train model
    model = train_model(X_train, y_train)

    # Check that the model has been trained correctly
    assert isinstance(model, LogisticRegression)


def test_evaluate_model():
    # Load sample data
    input_data = pd.read_csv("input_data.csv")

    # Clean data
    cleaned_data = clean_data(input_data)

    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(
        cleaned_data.drop("target", axis=1),
        cleaned_data["target"],
        test_size=0.2,
        random_state=42,
    )

    # Train model
    model = train_model(X_train, y_train)

    # Evaluate model
    accuracy = evaluate_model(model, X_test, y_test)

    # Check that the model has been evaluated correctly
    assert accuracy > 0.5
