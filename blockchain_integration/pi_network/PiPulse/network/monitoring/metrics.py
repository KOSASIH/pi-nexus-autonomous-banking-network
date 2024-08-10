class MetricTracker:
    def __init__(self):
        self.metrics = {}

    def track_metric(self, name, value):
        self.metrics[name] = value

    def get_metric(self, name):
        return self.metrics.get(name)

    def get_all_metrics(self):
        return self.metrics

    def reset_metrics(self):
        self.metrics = {}
