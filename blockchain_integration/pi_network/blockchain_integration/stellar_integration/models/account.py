from dataclasses import dataclass
from stellar_sdk import Account

@dataclass
class StellarAccount(StellarModel):
    account_id: str
    sequence_number: int
    subentry_count: int

    @classmethod
    def from_xdr(cls, xdr_data):
        account = Account.from_xdr(xdr_data)
        return cls(
            account_id=account.account_id,
            sequence_number=account.sequence_number,
            subentry_count=account.subentry_count
        )

    def to_xdr(self):
        return self.account_id.to_xdr()
