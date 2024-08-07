import os
import psutil
import ping3

class NetworkHealth:
    def __init__(self):
        self.nodes = ['node1', 'node2', 'node3']  # list of node names or IP addresses
        self.ping_results = {}

    def ping_nodes(self):
        for node in self.nodes:
            try:
                ping_time = ping3.ping(node, timeout=1)
                self.ping_results[node] = ping_time
            except Exception as e:
                self.ping_results[node] = str(e)

    def check_node_status(self):
        for node, ping_time in self.ping_results.items():
            if ping_time is not None:
                if ping_time > 100:
                    print(f"Node {node} is slow to respond ({ping_time} ms)")
                else:
                    print(f"Node {node} is online ({ping_time} ms)")
            else:
                print(f"Node {node} is offline ({ping_time})")

    def check_network_io(self):
        net_io = psutil.net_io_counters()
        print(f"Network I/O: {net_io.bytes_sent} bytes sent, {net_io.bytes_recv} bytes received")

    def run(self):
        while True:
            self.ping_nodes()
            self.check_node_status()
            self.check_network_io()
            time.sleep(10)

if __name__ == '__main__':
    network_health = NetworkHealth()
    network_health.run()
