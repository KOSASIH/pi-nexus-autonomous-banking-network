// File name: real_time_data_analytics.scala
import org.apache.spark.SparkConf
import org.apache.spark.streaming.{Seconds, StreamingContext}
import org.apache.spark.streaming.kafka010.KafkaUtils
import org.apache.kafka.clients.consumer.ConsumerRecord

object RealTimeDataAnalytics {
  def main(args: Array[String]) {
    val sparkConf = new SparkConf().setAppName("Real-Time Data Analytics")
    val ssc = new StreamingContext(sparkConf, Seconds(10))

    val kafkaParams = Map[String, String](
      "bootstrap.servers" -> "localhost:9092",
      "key.deserializer" -> "org.apache.kafka.common.serialization.StringDeserializer",
      "value.deserializer" -> "org.apache.kafka.common.serialization.StringDeserializer"
    )

    val kafkaStream = KafkaUtils.createDirectStream[String, String](
      ssc,
      LocationStrategies.PreferConsistent,
      ConsumerStrategies.Subscribe[String, String](Array("drone_data"), kafkaParams)
    )

    kafkaStream.foreachRDD(rdd => {
      rdd.foreach(record => {
        val data = record.value().split(",")
        val drone_id = data(0)
        val sensor_data = data(1)
        val timestamp = data(2)

        // Perform real-time data analytics here
        println(s"Drone $drone_id: $sensor_data at $timestamp")
      })
    })

    ssc.start()
    ssc.awaitTermination()
  }
}
