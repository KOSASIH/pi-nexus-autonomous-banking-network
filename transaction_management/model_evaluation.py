import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix

def evaluate_transaction_model_performance(model, X_test, y_test):
    """
    Evaluates the performance of the trained transaction machine learning model using various metrics.
    """
    y_pred = model.predict(X_test)
    report = classification_report(y_test, y_pred)
    matrix = confusion_matrix(y_test, y_pred)
    df = pd.DataFrame(matrix, columns=['Predicted No', 'Predicted Yes'], index=['Actual No', 'Actual Yes'])
    return report, df
