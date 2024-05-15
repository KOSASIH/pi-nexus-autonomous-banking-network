# pi_nexus/transactions.py
class Transaction:
    def __init__(self, id: int, data: dict) -> None:
        self.id = id
        self.amount = data["amount"]


class Account:
    def __init__(self, id: int, data: dict) -> None:
        self.id = id
        self.balance = data["balance"]
