import pandas as pd
from sklearn.preprocessing import LabelEncoder
from ml.models.riskAssessment.risk_assessment_model import RiskAssessmentModel

def load_data(file_path):
    data = pd.read_csv(file_path)
    X = data.drop(['risk_score'], axis=1)
    y = data['risk_score']
    le = LabelEncoder()
    y = le.fit_transform(y)
    return X, y

def train_model(X, y):
    model = RiskAssessmentModel()
    model.train(X, y)

def main():
    file_path = 'ml/data/risk_assessment_data.csv'
    X, y = load_data(file_path)
    train_model(X, y)

if __name__ == '__main__':
    main()
