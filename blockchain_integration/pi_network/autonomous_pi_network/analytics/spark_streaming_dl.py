from pyspark.sql import SparkSession
from pyspark.ml import Pipeline
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.regression import LinearRegression
from pyspark.ml.evaluation import RegressionEvaluator

class SparkStreamingDL:
    def __init__(self, spark_session):
        self.spark_session = spark_session

    defcreate_pipeline(self):
        # Create a pipeline for real-time data analytics
        assembler = VectorAssembler(inputCols=["feature1", "feature2"], outputCol="features")
        lr = LinearRegression(featuresCol="features", labelCol="target")
        pipeline = Pipeline(stages=[assembler, lr])
        return pipeline

    def train_model(self, data):
        # Train a deep learning model on the real-time data stream
        pipeline = self.create_pipeline()
        model = pipeline.fit(data)
        return model

    def make_predictions(self, model, data):
        # Make predictions on the real-time data stream using the trained model
        predictions = model.transform(data)
        return predictions

    def evaluate_model(self, predictions):
        # Evaluate the performance of the deep learning model
        evaluator = RegressionEvaluator(
            labelCol="target", predictionCol="prediction", metricName="rmse"
        )
        rmse = evaluator.evaluate(predictions)
        return rmse
