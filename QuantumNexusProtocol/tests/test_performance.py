import unittest
from src.performance.network_monitor import NetworkMonitor

class TestPerformance(unittest.TestCase):
    def setUp(self):
        self.monitor = NetworkMonitor()

    def test_latency(self):
        latency = self.monitor.measure_latency()
        self.assertLess(latency, 100)  # Assuming latency should be less than 100ms

    def test_throughput(self):
        throughput = self.monitor.measure_throughput()
        self.assertGreater(throughput, 1000)  # Assuming throughput should be greater than 1000 transactions per second

if __name__ == '__main__':
    unittest.main()
