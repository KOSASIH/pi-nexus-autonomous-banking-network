import datetime
import time

class TransactionScheduler:
    def __init__(self):
        self.scheduled_transactions = []

    def schedule_transaction(self, transaction, timestamp):
        self.scheduled_transactions.append((transaction, timestamp))

    def process_scheduled_transactions(self):
        while True:
            current_timestamp = int(time.time())
            for transaction, timestamp in self.scheduled_transactions:
                if timestamp <= current_timestamp:
                    # Process the transaction
                    print(f"Processing transaction: {transaction}")
                    self.scheduled_transactions.remove((transaction, timestamp))
            time.sleep(1)

# Example usage:
scheduler = TransactionScheduler()

transaction1 = "Send 1 BTC to John"
timestamp1 = int(time.time()) + 60  # Schedule in 1 minute
scheduler.schedule_transaction(transaction1, timestamp1)

transaction2 = "Send 0.5 ETH to Jane"
timestamp2 = int(time.time()) + 120  # Schedule in 2 minutesscheduler.schedule_transaction(transaction2, timestamp2)

scheduler.process_scheduled_transactions()
