class IssueDetection:
    def __init__(self, system_monitoring):
        self.system_monitoring = system_monitoring

    def detect_issues(self):
        if self.system_monitoring.cpu_percent > 80:
            print('High CPU usage detected!')
        if self.system_monitoring.memory_percent > 80:
            print('High memory usage detected!')
        if self.system_monitoring.disk_percent > 80:
            print('High disk usage detected!')
