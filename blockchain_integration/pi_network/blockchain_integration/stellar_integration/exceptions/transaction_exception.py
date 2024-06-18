# transaction_exception.py
import time
import random

class TransactionException(StellarException):
    def __init__(self, message, code, data=None, max_retries=3, backoff_factor=2):
        super().__init__(message, code, data)
        self.max_retries = max_retries
        self.backoff_factor = backoff_factor
        self.retry_count = 0

    def retry_transaction(self, func, *args, **kwargs):
        while self.retry_count < self.max_retries:
            try:
                return func(*args, **kwargs)
            except StellarException as e:
                self.retry_count += 1
                backoff_time = self.backoff_factor ** self.retry_count
                time.sleep(backoff_time + random.uniform(0, 1))
                print(f"Retrying transaction ({self.retry_count}/{self.max_retries}) in {backoff_time} seconds")
        raise self
