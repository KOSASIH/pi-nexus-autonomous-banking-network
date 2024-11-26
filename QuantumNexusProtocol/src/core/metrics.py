import json
from datetime import datetime, timedelta

class Metrics:
    def __init__(self):
        self.metrics_data = {
            'transaction_count': 0,
            'block_count': 0,
            'block_times': [],
            'last_block_time': None
        }

    def record_transaction(self):
        """Record a new transaction."""
        self.metrics_data['transaction_count'] += 1
        print(f"Transaction recorded. Total transactions: {self.metrics_data['transaction_count']}")

    def record_block(self):
        """Record a new block and its time."""
        current_time = datetime.utcnow()
        if self.metrics_data['last_block_time']:
            block_time = (current_time - self.metrics_data['last_block_time']).total_seconds()
            self.metrics_data['block_times'].append(block_time)
            print(f"Block mined. Time taken: {block_time:.2f} seconds")

        self.metrics_data['block_count'] += 1
        self.metrics_data['last_block_time'] = current_time
        print(f"Total blocks: {self.metrics_data['block_count']}")

    def average_block_time(self):
        """Calculate the average time taken to mine a block."""
        if not self.metrics_data['block_times']:
            return 0
        return sum(self.metrics_data['block_times']) / len(self.metrics_data['block_times'])

    def get_metrics(self):
        """Return the current metrics data."""
        return {
            'transaction_count': self.metrics_data['transaction_count'],
            'block_count': self.metrics_data['block_count'],
            'average_block_time': self.average_block_time(),
            'last_block_time': self.metrics_data['last_block_time'].isoformat() if self.metrics_data['last_block_time'] else None
        }

    def save_to_file(self, filename):
        """Save metrics data to a JSON file."""
        with open(filename, 'w') as f:
            json.dump(self.get_metrics(), f, indent=4)
        print(f"Metrics saved to {filename}")

    def load_from_file(self, filename):
        """Load metrics data from a JSON file."""
        try:
            with open(filename, 'r') as f:
                self.metrics_data = json.load(f)
            # Convert last_block_time back to datetime
            if self.metrics_data['last_block_time']:
                self.metrics_data['last_block_time'] = datetime.fromisoformat(self.metrics_data['last_block_time'])
            print(f"Metrics loaded from {filename}")
        except FileNotFoundError:
            print(f"File {filename} not found.")
        except json.JSONDecodeError:
            print(f"Error decoding JSON from the file {filename}.")

# Example usage
if __name__ == '__main__':
    metrics = Metrics()

    # Simulate recording transactions and blocks
    metrics.record_transaction()
    metrics.record_transaction()
    metrics.record_block()
    metrics.record_block()

    # Retrieve and print metrics
    current_metrics = metrics.get_metrics()
    print("Current Metrics:", current_metrics)

    # Save metrics to a file
    metrics.save_to_file('metrics.json')

    # Load metrics from a file
    metrics.load_from_file('metrics.json')
