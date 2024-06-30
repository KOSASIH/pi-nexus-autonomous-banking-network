import os
from kafka import KafkaProducer
from confluent_kafka import avro

class TransactionProcessing:
    def __init__(self):
        self.producer = KafkaProducer(bootstrap_servers='localhost:9092')

    def process_transaction(self, transaction_data):
        # Produce transaction data to Kafka topic
        self.producer.send('transactions', value=transaction_data)
