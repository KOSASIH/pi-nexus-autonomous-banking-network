package com.sidra;

import org.apache.kafka.clients.consumer.ConsumerConfig;
import org.apache.kafka.clients.consumer.ConsumerRecord;
import org.apache.kafka.clients.consumer.ConsumerRecords;
import org.apache.kafka.clients.consumer.KafkaConsumer;
import org.apache.kafka.common.serialization.StringDeserializer;
import org.apache.spark.SparkConf;
import org.apache.spark.api.java.JavaSparkContext;
import org.apache.spark.streaming.Duration;
import org.apache.spark.streaming.api.java.JavaStreamingContext;

import java.util.Collections;
import java.util.Properties;

public class RealtimeAnalytics {
    public static void main(String[] args) {
        // Set up a Kafka consumer
        Properties props = new Properties();
        props.put("bootstrap.servers", "localhost:9092");
        props.put("group.id", "test");
        props.put("key.deserializer", StringDeserializer.class.getName());
        props.put("value.deserializer", StringDeserializer.class.getName());

        KafkaConsumer<String, String> consumer = new KafkaConsumer<>(props);
        consumer.subscribe(Collections.singleton("topic"));

        // Set up a Spark context
        SparkConf conf = new SparkConf().setAppName("Realtime Analytics").setMaster("local[2]");
        JavaSparkContext sc = new JavaSparkContext(conf);
        JavaStreamingContext ssc = new JavaStreamingContext(sc, new Duration(1000));

        // Create a direct stream from Kafka
        ssc.receiverStream(new KafkaReceiver("localhost:9092", "test", "topic"))
                .foreachRDD(rdd -> {
                    // Process the RDD
                    rdd.foreach(record -> {
                        // Analyze the record
                        System.out.println("Received message: " + record);
                    });
                });

        // Start the streaming context
        ssc.start();
        ssc.awaitTermination();
    }
}
