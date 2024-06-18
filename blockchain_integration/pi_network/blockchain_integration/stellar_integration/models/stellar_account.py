# stellar_account.py
from stellar_sdk.account import Account
from stellar_sdk.exceptions import StellarSdkError
from cachetools import cached, TTLCache

class StellarAccount(Account):
    def __init__(self, account_id, sequence, *args, **kwargs):
        super().__init__(account_id, sequence, *args, **kwargs)
        self.cache = TTLCache(maxsize=100, ttl=300)  # 5-minute cache

    @cached(cache)
    def get_balance(self, asset_code):
        try:
            return super().get_balance(asset_code)
        except StellarSdkError as e:
            raise StellarAccountError(f"Failed to get balance for {asset_code}: {e}")

    def validate(self):
        if not self.account_id:
            raise StellarAccountError("Account ID is required")
        if not self.sequence:
            raise StellarAccountError("Sequence number is required")
        # Add more validation rules as needed
