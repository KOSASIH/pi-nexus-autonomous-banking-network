# eonix.py
from blockchain import Blockchain
from wallet import Wallet
from transaction import Transaction
from contract import EonixContract
from database import EonixDatabase

class Eonix:
    def __init__(self):
        self.blockchain = Blockchain()
        self.wallet = Wallet()
        self.contract_manager = EonixContractManager()
        self.database = EonixDatabase()

    def create_transaction(self, recipient, amount):
        transaction = Transaction(self.wallet.public_key, recipient, amount)
        self.blockchain.add_transaction(transaction)

    def mine_block(self):
        self.blockchain.mine_pending_transactions()

    def get_balance(self):
        balance = 0
        for block in self.blockchain.chain:
            for transaction in block.transactions:
                if transaction.recipient == self.wallet.public_key:
                    balance += transaction.amount
                elif transaction.sender == self.wallet.public_key:
                    balance -= transaction.amount
        return balance

    def execute_contract(self, code, inputs):
        contract = EonixContract(code)
        return contract.execute(inputs)

    def store_data(self, data):
        return self.database.store_data(data)

    def retrieve_data(self, cid):
        return self.database.retrieve_data(cid)
