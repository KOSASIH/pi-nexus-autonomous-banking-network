import xgboost as xgb

class ModelEvaluation:
    def __init__(self, model_file):
        self.model_file = model_file

    def evaluate_model(self, data, target_column):
        """
        Evaluates the performance of the trained model.
        """
        model = xgb.Booster(model_file=self.model_file)

        X = data.drop(target_column, axis=1)
        y = data[target_column]

        dtest = xgb.DMatrix(X, label=y)

        y_pred = model.predict(dtest)

        rmse = (sum((y_pred - y)**2) / len(y))**0.5

        return rmse
