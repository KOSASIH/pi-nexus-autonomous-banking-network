import web3
from web3.contract import Contract

class ERC725IdentityManager:
    def __init__(self, contract_address, private_key):
        self.contract_address = contract_address
        self.private_key = private_key
        self.contract = Contract(self.contract_address, abi=ERC725_ABI)

    def create_identity(self, user_data):
        tx_hash = self.contract.functions.createIdentity(user_data).transact({'from': self.private_key})
        return tx_hash

    def update_identity(self, user_data):
        tx_hash = self.contract.functions.updateIdentity(user_data).transact({'from': self.private_key})
        return tx_hash

# Example usage:
erc725_contract_address = '0x...'
private_key = '0x...'
erc725_identity_manager = ERC725IdentityManager(erc725_contract_address, private_key)
user_data = {'name': 'John Doe', 'email': 'johndoe@example.com'}
tx_hash = erc725_identity_manager.create_identity(user_data)
print(tx_hash)
