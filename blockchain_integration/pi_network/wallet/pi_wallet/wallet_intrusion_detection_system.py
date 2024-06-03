import json
import os
import subprocess


class IntrusionDetectionSystem:
    def __init__(self, log_path):
        self.log_path = log_path

    def monitor_logs(self):
        # Monitor logs for suspicious activity
        while True:
            log_output = subprocess.check_output(["tail", "-f", self.log_path])
            log_json = json.loads(log_output)
            for log_entry in log_json:
                if self.is_suspicious(log_entry):
                    self.alert(log_entry)

    def is_suspicious(self, log_entry):
        # Implement logic to determine if log entry is suspicious
        # For demonstration purposes, assume log entry is suspicious if it contains the word "error"
        return "error" in log_entry["message"]

    def alert(self, log_entry):
        # Implement logic to alert security team of suspicious activity
        # For demonstration purposes, print a message to the console
        print(f"Suspicious activity detected: {log_entry['message']}")


if __name__ == "__main__":
    log_path = "/path/to/logs"
    intrusion_detection_system = IntrusionDetectionSystem(log_path)
    intrusion_detection_system.monitor_logs()
