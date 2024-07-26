# sidra_chain_data_streaming.py
import kafka
from kafka import KafkaProducer

class SidraChainDataStreaming:
    def __init__(self, bootstrap_servers):
        self.bootstrap_servers = bootstrap_servers
        self.producer = KafkaProducer(bootstrap_servers=self.bootstrap_servers)

    def produce_data(self, topic, data):
        # Produce data to a Kafka topic
        self.producer.send(topic, value=data.encode())
        self.producer.flush()

    def consume_data(self, topic):
        # Consume data from a Kafka topic
        consumer = kafka.KafkaConsumer(topic, bootstrap_servers=self.bootstrap_servers)
        for message in consumer:
            yield message.value.decode()
