# Improved code with dataclasses and type hints
from dataclasses import dataclass
from typing import List

@dataclass
class BankAccount:
    account_number: str
    account_holder: str
    balance: float
    transactions: List[dict]

    def __post_init__(self):
        self.transactions = []
