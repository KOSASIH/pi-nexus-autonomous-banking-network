import xgboost as xgb
from sklearn.model_selection import train_test_split

class ModelTraining:
    def __init__(self, data, target_column):
        self.data = data
        self.target_column = target_column

    def train_model(self, model_file):
        """
        Trains the machine learning model using the prepared data.
        """
        X = self.data.drop(self.target_column, axis=1)
        y = self.data[self.target_column]

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        dtrain = xgb.DMatrix(X_train, label=y_train)
        dtest = xgb.DMatrix(X_test, label=y_test)

        params = {
            'objective': 'reg:squarederror',
            'eval_metric': 'rmse',
            'max_depth': 6,
            'eta': 0.1,
            'subsample': 0.8,
            'colsample_bytree': 0.8
        }

        model = xgb.train(params, dtrain, num_boost_round=1000, evals=[(dtest, 'test')], early_stopping_rounds=50)

        model.save_model(model_file)

        return model
