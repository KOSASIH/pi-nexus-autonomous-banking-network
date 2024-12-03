class Transaction:
    def __init__(self, wallet_manager):
        self.wallet_manager = wallet_manager

    def send(self, sender, receiver, amount):
        if sender not in self.wallet_manager.wallets:
            raise ValueError("Sender wallet does not exist.")
        if receiver not in self.wallet_manager.wallets:
            raise ValueError("Receiver wallet does not exist.")
        if self.wallet_manager.get_balance(sender) < amount:
            raise ValueError("Insufficient balance for the transaction.")

        # Process the transaction
        self.wallet_manager.withdraw(sender, amount)
        self.wallet_manager.deposit(receiver, amount)
        self.wallet_manager.transaction_history[sender].append(f"Sent {amount} to {receiver}.")
        self.wallet_manager.transaction_history[receiver].append(f"Received {amount} from {sender}.")
        print(f"Transaction successful: {amount} sent from {sender} to {receiver}.")

# Example usage
if __name__ == "__main__":
    wallet_manager = Wallet()
    transaction_manager = Transaction(wallet_manager)

    # Create wallets for users
    wallet_manager.create_wallet("user1", "seed_phrase_1")
    wallet_manager.create_wallet("user2", "seed_phrase_2")

    # Deposit funds
    wallet_manager.deposit("user1", 100)

    # Send funds from user1 to user2
    transaction_manager.send("user1", "user2", 30)

    # Check balances and transaction history
    print(f"User 1 balance: {wallet_manager.get_balance('user1')}")
    print(f"User 2 balance: {wallet_manager.get_balance('user2')}")
    print(wallet_manager.get_wallet_info("user1"))
    print(wallet_manager.get_wallet_info("user2"))
