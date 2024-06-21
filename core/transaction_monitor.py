import os
import json
from kafka import KafkaConsumer
from elasticsearch import Elasticsearch

class TransactionMonitor:
  def __init__(self, kafka_bootstrap_servers, elasticsearch_url):
    self.kafka_consumer = KafkaConsumer('transactions', bootstrap_servers=kafka_bootstrap_servers)
    self.elasticsearch = Elasticsearch([elasticsearch_url])

  def start_monitoring(self):
    for message in self.kafka_consumer:
      transaction = json.loads(message.value)
      self.elasticsearch.index(index='transactions', body=transaction)

if __name__ == '__main__':
  kafka_bootstrap_servers = 'localhost:9092'
  elasticsearch_url = 'http://localhost:9200'
  monitor = TransactionMonitor(kafka_bootstrap_servers, elasticsearch_url)
  monitor.start_monitoring()
