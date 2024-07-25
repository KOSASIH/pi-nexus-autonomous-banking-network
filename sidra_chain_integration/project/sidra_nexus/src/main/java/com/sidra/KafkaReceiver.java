package com.sidra;

import kafka.consumer.KafkaStream;
import kafka.message.MessageAndMetadata;
import org.apache.spark.streaming.receiver.Receiver;

public class KafkaReceiver extends Receiver<String> {
    private KafkaStream<String, String> kafkaStream;

    public KafkaReceiver(String kafkaBroker, String groupId, String topic) {
        // Set up the Kafka stream
        Properties props = new Properties();
        props.put("bootstrap.servers", kafkaBroker);
        props.put("group.id", groupId);

        kafkaStream = KafkaUtils.createStream(props, topic);
    }

    @Override
    public void onStart() {
        // Start the Kafka stream
        kafkaStream.start();
    }

    @Override
    public void onStop() {
        // Stop the Kafka stream
        kafkaStream.stop();
    }

    @Override
    public void onReceive(Object obj) {
        // Process the message
        MessageAndMetadata<String, String> message = (MessageAndMetadata<String, String>) obj;
        store(message.message());
    }
}
