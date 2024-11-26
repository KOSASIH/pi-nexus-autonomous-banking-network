import json
from datetime import datetime

class EventLog:
    def __init__(self):
        self.logs = []

    def log_event(self, event_type, details):
        """Log an event with a timestamp."""
        timestamp = datetime.utcnow().isoformat()
        event = {
            'timestamp': timestamp,
            'event_type': event_type,
            'details': details
        }
        self.logs.append(event)
        print(f"Logged event: {event}")  # Optional: Print to console for immediate feedback

    def get_logs(self):
        """Return all logs."""
        return self.logs

    def filter_logs(self, event_type=None, start_time=None, end_time=None):
        """Filter logs based on event type and time range."""
        filtered_logs = self.logs

        if event_type:
            filtered_logs = [log for log in filtered_logs if log['event_type'] == event_type]

        if start_time:
            filtered_logs = [log for log in filtered_logs if log['timestamp'] >= start_time]

        if end_time:
            filtered_logs = [log for log in filtered_logs if log['timestamp'] <= end_time]

        return filtered_logs

    def save_to_file(self, filename):
        """Save logs to a JSON file."""
        with open(filename, 'w') as f:
            json.dump(self.logs, f, indent=4)
        print(f"Logs saved to {filename}")

    def load_from_file(self, filename):
        """Load logs from a JSON file."""
        try:
            with open(filename, 'r') as f:
                self.logs = json.load(f)
            print(f"Logs loaded from {filename}")
        except FileNotFoundError:
            print(f"File {filename} not found.")
        except json.JSONDecodeError:
            print(f"Error decoding JSON from the file {filename}.")

# Example usage
if __name__ == '__main__':
    event_log = EventLog()

    # Log some events
    event_log.log_event('TRANSACTION_CREATED', {'sender': 'Alice', 'recipient': 'Bob', 'amount': 50})
    event_log.log_event('BLOCK_MINED', {'block_index': 1, 'transactions': 1})

    # Retrieve all logs
    all_logs = event_log.get_logs()
    print("All Logs:", all_logs)

    # Filter logs
    filtered_logs = event_log.filter_logs(event_type='TRANSACTION_CREATED')
    print("Filtered Logs (TRANSACTION_CREATED):", filtered_logs)

    # Save logs to a file
    event_log.save_to_file('event_logs.json')

    # Load logs from a file
    event_log.load_from_file('event_logs.json')
