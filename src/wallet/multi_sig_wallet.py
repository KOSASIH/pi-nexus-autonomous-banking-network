from collections import defaultdict

class MultiSigWallet:
    def __init__(self, required_signatures):
        self.required_signatures = required_signatures
        self.signatures = defaultdict(list)  # transaction_id -> list of signers
        self.transactions = []  # List of transactions

    def create_transaction(self, transaction_id, sender, receiver, amount):
        transaction = {
            "transaction_id": transaction_id,
            "sender": sender,
            "receiver": receiver,
            "amount": amount,
            "executed": False
        }
        self.transactions.append(transaction)
        return transaction

    def sign_transaction(self, transaction_id, signer):
        for transaction in self.transactions:
            if transaction["transaction_id"] == transaction_id and not transaction["executed"]:
                if signer not in self.signatures[transaction_id]:
                    self.signatures[transaction_id].append(signer)
                    print(f"{signer} signed transaction {transaction_id}.")
                    if len(self.signatures[transaction_id]) >= self.required_signatures:
                        self.execute_transaction(transaction_id)
                return
        print("Transaction not found or already executed.")

    def execute_transaction(self, transaction_id):
        for transaction in self.transactions:
            if transaction["transaction_id"] == transaction_id and not transaction["executed"]:
                transaction["executed"] = True
                print(f"Transaction {transaction_id} executed: {transaction['amount']} from {transaction['sender']} to {transaction['receiver']}.")
                return
        print("Transaction not found or already executed.")
