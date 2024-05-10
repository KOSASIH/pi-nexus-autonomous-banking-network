import hashlib
import time

from transaction import Transaction


class Contract:
    def __init__(self, address, code):
        self.address = address
        self.code = code
        self.state = {}

    def execute(self, transaction):
        # Check if the transaction sender has enough funds
        if self.state.get(transaction.sender, 0) < transaction.amount:
            return False

        # Execute the contract code
        code_output = eval(self.code)

        # Update the contract state
        self.state[transaction.receiver] = (
            self.state.get(transaction.receiver, 0) + transaction.amount
        )
        self.state[transaction.sender] = (
            self.state.get(transaction.sender, 0) - transaction.amount
        )

        # Return the output of the contract code
        return code_output

    def to_dict(self):
        return {"address": self.address, "code": self.code, "state": self.state}
