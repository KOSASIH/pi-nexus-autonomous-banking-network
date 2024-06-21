import stellar_sdk

class StellarIntegration:
    def __init__(self):
        self.stellar_sdk = stellar_sdk.Client()

    def get_account_balance(self, account_id):
        try:
            account = self.stellar_sdk.account(account_id)
            balance = account.balances[0].balance
            return balance
        except stellar_sdk.exceptions.NotFoundError:
            logging.error(f"Account not found: {account_id}")
            return None
        except stellar_sdk.exceptions.RequestError as e:
            logging.error(f"Error fetching account balance: {e}")
            return None

    def send_transaction(self, source_account_id, dest_account_id, amount, memo):
        try:
            tx = self.stellar_sdk.Transaction(
                source_account=source_account_id,
                destination_account=dest_account_id,
                amount=amount,
                memo=memo
            )
            tx_hash = tx.hash
            return tx_hash
        except stellar_sdk.exceptions.RequestError as e:
            logging.error(f"Error sending transaction: {e}")
            return None

    def get_transaction_history(self, account_id):
        try:
            tx_history = self.stellar_sdk.transactions(account_id)
            return tx_history
        except stellar_sdk.exceptions.RequestError as e:
            logging.error(f"Error fetching transaction history: {e}")
            return None
