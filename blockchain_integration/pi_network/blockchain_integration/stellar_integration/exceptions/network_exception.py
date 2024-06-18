# network_exception.py
import time

class NetworkException(StellarException):
    def __init__(self, message, code, data=None, threshold=5, window=30):
        super().__init__(message, code, data)
        self.threshold = threshold
        self.window = window
        self.failure_count = 0
        self.last_failure_time = 0

    def is_circuit_open(self):
        if self.failure_count >= self.threshold:
            if time.time() - self.last_failure_time > self.window:
                self.failure_count = 0
                return False
            return True
        return False

    def report_failure(self):
        self.failure_count += 1
        self.last_failure_time = time.time()

    def __str__(self):
        return f"{self.code}: {self.message} (Circuit open: {self.is_circuit_open()})"
