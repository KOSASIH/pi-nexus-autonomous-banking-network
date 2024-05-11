import time
import prometheus_client

class Monitor:
    def __init__(self):
        self.counter = prometheus_client.Counter('transaction_counter', 'Number of transactions processed')
        self.gauge = prometheus_client.Gauge('transaction_latency_seconds', 'Latency of transactions in seconds')

    def monitor_transaction(self, transaction_time):
        """
        Monitors a transaction and records its latency.
        """
        self.counter.inc()
        self.gauge.set(transaction_time)

    def start_monitoring(self):
        """
        Starts the monitoring system.
        """
        prometheus_client.start_http_server(8000)
        while True:
            time.sleep(1)
