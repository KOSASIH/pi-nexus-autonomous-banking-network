class TransactionRouter:
    def __init__(self):
        self.routing_rules = {
            "domestic": {
                "regex": r"^\d{10}$",
                "message": "Invalid account number format. Please enter a 10-digit account number.",
            },
            "international": {
                "regex": r"^\d{12}$",
                "message": "Invalid account number format. Please enter a 12-digit account number.",
            },
        }

    def route_transaction(self, transaction):
        for rule in self.routing_rules:
            if re.match(
                self.routing_rules[rule]["regex"], transaction["account_number"]
            ):
                return rule
        raise ValueError("Invalid account number format.")
