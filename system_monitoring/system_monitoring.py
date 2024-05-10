import psutil
import time

class SystemMonitoring:
    def __init__(self):
        self.cpu_percent = psutil.cpu_percent()
        self.memory_percent = psutil.virtual_memory().percent
        self.disk_percent = psutil.disk_usage('/').percent

    def monitor_system(self):
        while True:
            time.sleep(60)
            self.cpu_percent = psutil.cpu_percent()
            self.memory_percent = psutil.virtual_memory().percent
            self.disk_percent = psutil.disk_usage('/').percent
            print(f'CPU: {self.cpu_percent}%')
            print(f'Memory: {self.memory_percent}%')
            print(f'Disk: {self.disk_percent}%')
