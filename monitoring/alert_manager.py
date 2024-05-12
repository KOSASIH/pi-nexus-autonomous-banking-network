import os
import time

from alertmanager import AlertManager
from prometheus_client import generate_latest, parse_textfile


class AlertManager:
    def __init__(self, prometheus_rulefile):
        self.prometheus_rulefile = prometheus_rulefile
        self.alertmanager = AlertManager()

    def start_alert_manager(self):
        """
        Starts the alert manager.
        """
        self.alertmanager.start()

        while True:
            # Fetch the latest metrics from Prometheus
            metrics = generate_latest()

            # Load the Prometheus rulefile
            with open(self.prometheus_rulefile, "r") as f:
                rules = f.read()

            # Evaluate the rules against the metrics
            alerts = parse_textfile(rules, metrics)

            # Send the alerts to the alert manager
            self.alertmanager.send(alerts)

            # Sleep for a while before checking again
            time.sleep(60)

    def stop_alert_manager(self):
        """
        Stops the alert manager.
        """
        self.alertmanager.stop()
