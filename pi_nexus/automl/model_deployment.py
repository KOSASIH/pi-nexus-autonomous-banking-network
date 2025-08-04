# automl/model_deployment.py
import joblib

from .model_training import ModelTraining


class ModelDeployment:
    def __init__(self):
        self.model_training = ModelTraining()

    def deploy_model(self, X, y):
        model = self.model_training.train_model(X, y)
        joblib.dump(model, "model.joblib")
