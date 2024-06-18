from dataclasses import dataclass
from typing import Optional

@dataclass
class Transaction:
    id: int
    coin_id: int
    amount: float
    timestamp: str

    def __str__(self) -> str:
        return f"Transaction {self.id}: {self.amount} {self.coin_id} at {self.timestamp}"
