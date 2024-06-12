from pyspark.sql import SparkSession
from pyspark.sql.functions import col, when

class SparkDataAnalytics:
    def __init__(self, spark_session):
        self.spark_session = spark_session

    def analyze_data(self, data):
        df = self.spark_session.createDataFrame(data)
        # Perform advanced data analytics using Spark SQL and Graph Processing
        result = df.groupBy('category').agg({'amount': 'um'}).collect()
        return result

# Example usage:
spark_session = SparkSession.builder.appName('PI-Nexus Data Analytics').getOrCreate()
spark_data_analytics = SparkDataAnalytics(spark_session)
data = [{'category': 'withdrawal', 'amount': 100}, {'category': 'deposit', 'amount': 50}]
result = spark_data_analytics.analyze_data(data)
print(result)
