# error_codes.py
from enum import Enum

class ErrorCodes(Enum):
    INVALID_REQUEST = (400, "Invalid request payload")
    INSUFFICIENT_BALANCE = (402, "Insufficient balance for transaction")
    TRANSACTION_FAILED = (500, "Transaction failed due to internal error")
    # Add more error codes as needed

    def __init__(self, code, message):
        self.code = code
        self.message = message

    def __str__(self):
        return f"{self.code}: {self.message}"
