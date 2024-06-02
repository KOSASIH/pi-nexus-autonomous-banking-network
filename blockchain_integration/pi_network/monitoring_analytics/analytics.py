import pandas as pd
import matplotlib.pyplot as plt

class Analytics:
    def __init__(self, data):
        self.data = data

    def generate_node_distribution(self):
        # Generate a bar chart showing the distribution of nodes
        node_counts = self.data['nodes'].value_counts()
        plt.bar(node_counts.index, node_counts.values)
        plt.xlabel('Node Type')
        plt.ylabel('Count')
        plt.title('Node Distribution')
        plt.show()

    def generate_performance_trend(self):
        # Generate a line chart showing the performance trend
        performance_data = self.data['performance']
        plt.plot(performance_data['timestamp'], performance_data['throughput'])
        plt.xlabel('Time')
        plt.ylabel('Throughput')
        plt.title('Performance Trend')
        plt.show()

    def generate_disk_usage_trend(self):
        # Generate a line chart showing the disk usage trend
        disk_usage_data = self.data['disk_usage']
        plt.plot(disk_usage_data['timestamp'], disk_usage_data['usage'])
        plt.xlabel('Time')
        plt.ylabel('Disk Usage')
        plt.title('Disk Usage Trend')
        plt.show()

    def generate_cpu_usage_trend(self):
        # Generate a line chart showing the CPU usage trend
        cpu_usage_data = self.data['cpu_usage']
        plt.plot(cpu_usage_data['timestamp'], cpu_usage_data['usage'])
        plt.xlabel('Time')
        plt.ylabel('CPU Usage')
        plt.title('CPU Usage Trend')
        plt.show()

    def generate_memory_usage_trend(self):
        # Generate a line chart showing the memory usage trend
        memory_usage_data = self.data['memory_usage']
        plt.plot(memory_usage_data['timestamp'], memory_usage_data['usage'])
        plt.xlabel('Time')
        plt.ylabel('Memory Usage')
        plt.title('Memory Usage Trend')
        plt.show()
