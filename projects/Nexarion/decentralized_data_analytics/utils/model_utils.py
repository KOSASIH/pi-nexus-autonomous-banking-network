from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

def train_model(data, model):
    X = data.drop('target', axis=1)
    y = data['target']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    print('Model Evaluation:')
    print('Accuracy:', accuracy_score(y_test, y_pred))
    print('Classification Report:')
    print(classification_report(y_test, y_pred))
    return model

def evaluate_model(model, data):
    X = data.drop('target', axis=1)
    y = data['target']
    y_pred = model.predict(X)
    print('Model Evaluation:')
    print('Accuracy:', accuracy_score(y, y_pred))
    print('Classification Report:')
    print(classification_report(y, y_pred))
