class Transaction:
    def __init__(self, sender: str, receiver: str, amount: int):
        self.sender = sender
        self.receiver = receiver
        self.amount = amount


def validate_api_key(api_key: str) -> bool:
    # Validate the API key
    # This is a placeholder for the actual validation process
    return True
