import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

from config import get_config


def evaluate_model():
    # Load transformed data from CSV file
    input_file = get_config("data_processing.input_file")
    df = pd.read_csv(input_file)

    # Extract features and target variables
    X = df[["feature1", "feature2", "feature3"]]
    y = df["target"]

    # Load the trained model from the pickle file
    output_file = get_config("model_training.output_file")
    model = LogisticRegression.load(output_file)

    # Evaluate the model on the test set
    X_test = df[["feature1", "feature2", "feature3"]]
    y_test = df["target"]
    y_pred = model.predict(X_test)

    # Calculate evaluation metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)

    # Print evaluation metrics
    print(f"Accuracy: {accuracy:.3f}")
    print(f"Precision: {precision:.3f}")
    print(f"Recall: {recall:.3f}")
    print(f"F1 Score: {f1:.3f}")


if __name__ == "__main__":
    evaluate_model()
