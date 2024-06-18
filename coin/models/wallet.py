from dataclasses import dataclass
from typing import Optional

@dataclass
class Wallet:
    id: int
    user_id: int
    coin_id: int
    balance: float

    def __str__(self) -> str:
        return f"Wallet {self.id}: {self.balance} {self.coin_id} for user {self.user_id}"
