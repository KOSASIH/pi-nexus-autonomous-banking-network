import time
from collections import deque

class PerformanceOptimizer:
    def __init__(self, max_block_size=1000, batch_size=10):
        self.max_block_size = max_block_size  # Maximum number of transactions per block
        self.batch_size = batch_size  # Number of transactions to batch before processing
        self.transaction_queue = deque()  # Queue to hold incoming transactions
        self.processed_transactions = []  # List to hold processed transactions
        self.execution_time = []  # List to track execution times for performance analysis

    def add_transaction(self, transaction):
        """Add a transaction to the queue."""
        self.transaction_queue.append(transaction)
        print(f"Transaction added: {transaction}")

        # Process transactions if the batch size is reached
        if len(self.transaction_queue) >= self.batch_size:
            self.process_transactions()

    def process_transactions(self):
        """Process a batch of transactions."""
        start_time = time.time()
        batch = []

        while self.transaction_queue and len(batch) < self.max_block_size:
            batch.append(self.transaction_queue.popleft())

        # Simulate processing the batch of transactions
        self.processed_transactions.extend(batch)
        print(f"Processed batch of {len(batch)} transactions.")

        # Record execution time
        execution_duration = time.time() - start_time
        self.execution_time.append(execution_duration)
        print(f"Batch processing time: {execution_duration:.4f} seconds")

    def optimize_block_size(self):
        """Dynamically adjust the block size based on performance metrics."""
        if not self.execution_time:
            print("No execution time data available for optimization.")
            return

        average_time = sum(self.execution_time) / len(self.execution_time)
        print(f"Average processing time: {average_time:.4f} seconds")

        # Adjust block size based on average processing time
        if average_time < 0.1 and self.max_block_size < 2000:  # Example threshold
            self.max_block_size += 100
            print(f"Increasing max block size to {self.max_block_size}")
        elif average_time > 0.5 and self.max_block_size > 100:
            self.max_block_size -= 100
            print(f"Decreasing max block size to {self.max_block_size}")

    def get_processed_transactions(self):
        """Return the list of processed transactions."""
        return self.processed_transactions

    def get_performance_metrics(self):
        """Return performance metrics."""
        return {
            'max_block_size': self.max_block_size,
            'batch_size': self.batch_size,
            'average_execution_time': sum(self.execution_time) / len(self.execution_time) if self.execution_time else 0,
            'total_processed_transactions': len(self.processed_transactions)
        }

# Example usage
if __name__ == '__main__':
    optimizer = PerformanceOptimizer(max_block_size=1000, batch_size=5)

    # Simulate adding transactions
    for i in range(15):
        optimizer.add_transaction(f"Transaction {i + 1}")

    # Optimize block size based on performance
    optimizer.optimize_block_size()

    # Retrieve processed transactions and performance metrics
    processed_transactions = optimizer.get_processed_transactions()
    performance_metrics = optimizer.get_performance_metrics()

    print("Processed Transactions:", processed_transactions)
    print("Performance Metrics:", performance_metrics)
