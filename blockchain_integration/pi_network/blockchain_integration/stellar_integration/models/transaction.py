from dataclasses import dataclass
from stellar_sdk import Transaction

@dataclass
class StellarTransaction(StellarModel):
    transaction: Transaction

    @classmethod
    def from_xdr(cls, xdr_data):
        transaction = Transaction.from_xdr(xdr_data)
        return cls(transaction=transaction)

    def to_xdr(self):
        return self.transaction.to_xdr()
