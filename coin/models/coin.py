from dataclasses import dataclass
from typing import Optional

@dataclass
class Coin:
    id: int
    name: str
    symbol: str
    amount: float

    def __str__(self) -> str:
        return f"{self.name} ({self.symbol})"
