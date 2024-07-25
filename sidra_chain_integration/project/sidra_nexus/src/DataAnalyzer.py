package com.sidra.nexus;

import org.apache.spark.SparkConf;
import org.apache.spark.api.java.JavaSparkContext;
import org.apache.spark.sql.Dataset;
import org.apache.spark.sql.Row;
import org.apache.spark.sql.SparkSession;

public class DataAnalyzer {
    private SparkSession sparkSession;

    public DataAnalyzer() {
        // Set up a Spark session
        SparkConf conf = new SparkConf().setAppName("Data Analyzer").setMaster("local[2]");
        JavaSparkContext sc = new JavaSparkContext(conf);
        sparkSession = new SparkSession(sc.sc());
    }

    public void analyzeData(String filePath) {
        // Load data from a file
        Dataset<Row> data = sparkSession.read().format("csv").option("header", "true").load(filePath);

        // Perform data analysis
        data.groupBy("column1").count().show();
    }
}
