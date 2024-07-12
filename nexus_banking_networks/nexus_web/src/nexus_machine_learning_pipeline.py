import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


class NexusMachineLearningPipeline:

    def __init__(self):
        self.data = pd.read_csv("data.csv")
        self.X = self.data.drop("target", axis=1)
        self.y = self.data["target"]

    def create_pipeline(self):
        pipeline = Pipeline(
            [
                ("scaler", StandardScaler()),
                (
                    "classifier",
                    RandomForestClassifier(n_estimators=100, random_state=42),
                ),
            ]
        )
        return pipeline

    def train_pipeline(self, pipeline):
        pipeline.fit(self.X, self.y)

    def predict(self, pipeline, input_data):
        return pipeline.predict(input_data)
