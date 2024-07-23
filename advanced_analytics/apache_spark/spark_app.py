from pyspark.sql import SparkSession
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.regression import RandomForestRegressor

def spark_app():
    spark = SparkSession.builder.appName('PiNexusAnalytics').getOrCreate()
    data = spark.read.csv('data/transactions.csv', header=True, inferSchema=True)
    assembler = VectorAssembler(inputCols=['feature1', 'feature2'], outputCol='features')
    data = assembler.transform(data)
    model = RandomForestRegressor(featuresCol='features', labelCol='label')
    model.fit(data)
    return spark, model
