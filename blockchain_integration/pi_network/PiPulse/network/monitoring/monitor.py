import os
import time
import psutil
import GPUtil
import torch
from metrics import MetricTracker
from notifications import NotificationSender

class SystemMonitor:
    def __init__(self, interval=1, gpu_id=0):
        self.interval = interval
        self.gpu_id = gpu_id
        self.metric_tracker = MetricTracker()
        self.notification_sender = NotificationSender()

    def monitor_system(self):
        while True:
            self.monitor_gpu()
            self.monitor_cpu()
            self.monitor_memory()
            self.metric_tracker.track_metrics()
            time.sleep(self.interval)

    def monitor_gpu(self):
        gpu = GPUtil.getGPUs()[self.gpu_id]
        gpu_util = gpu.load * 100
        gpu_memory_util = gpu.memoryUtil * 100
        self.metric_tracker.track_metric('gpu_util', gpu_util)
        self.metric_tracker.track_metric('gpu_memory_util', gpu_memory_util)

    def monitor_cpu(self):
        cpu_util = psutil.cpu_percent()
        self.metric_tracker.track_metric('cpu_util', cpu_util)

    def monitor_memory(self):
        memory_util = psutil.virtual_memory().percent
        self.metric_tracker.track_metric('memory_util', memory_util)

    def send_notifications(self):
        if self.metric_tracker.get_metric('gpu_util') > 90:
            self.notification_sender.send_notification('GPU utilization is high!')
        if self.metric_tracker.get_metric('cpu_util') > 90:
            self.notification_sender.send_notification('CPU utilization is high!')
        if self.metric_tracker.get_metric('memory_util') > 90:
            self.notification_sender.send_notification('Memory utilization is high!')

if __name__ == '__main__':
    monitor = SystemMonitor()
    monitor.monitor_system()
