import json
from web3 import Web3

class IdentityManagement:
    def __init__(self, contract_address, abi, provider_url):
        self.web3 = Web3(Web3.HTTPProvider(provider_url))
        self.contract = self.web3.eth.contract(address=contract_address, abi=abi)

    def register_identity(self, account, private_key, name, email):
        nonce = self.web3.eth.getTransactionCount(account)
        tx = self.contract.functions.registerIdentity(name, email).buildTransaction({
            'chainId': 1,  # Mainnet ID, change as needed
            'gas': 2000000,
            'gasPrice': self.web3.toWei('50', 'gwei'),
            'nonce': nonce,
        })
        signed_tx = self.web3.eth.account.signTransaction(tx, private_key)
        tx_hash = self.web3.eth.sendRawTransaction(signed_tx.rawTransaction)
        return self.web3.toHex(tx_hash)

    def update_identity(self, account, private_key, name, email):
        nonce = self.web3.eth.getTransactionCount(account)
        tx = self.contract.functions.updateIdentity(name, email).buildTransaction({
            'chainId': 1,
            'gas': 2000000,
            'gasPrice': self.web3.toWei('50', 'gwei'),
            'nonce': nonce,
        })
        signed_tx = self.web3.eth.account.signTransaction(tx, private_key)
        tx_hash = self.web3.eth.sendRawTransaction(signed_tx.rawTransaction)
        return self.web3.toHex(tx_hash)

    def get_identity(self, user_address):
        return self.contract.functions.getIdentity(user_address).call()

    def identity_exists(self, user_address):
        return self.contract.functions.identityExists(user_address).call()

if __name__ == "__main__":
    # Example usage
    provider_url = "https://your.ethereum.node"  # Replace with your Ethereum node URL
    contract_address = "0xYourContractAddress"  # Replace with your deployed contract address
    abi = json.loads('[{"inputs":[{"internalType":"string","name":"_name","type":"string"},{"internalType":"string","name":"_email","type":"string"}],"name":"registerIdentity","outputs":[],"stateMutability":"nonpayable","type":"function"}, ...]')  # Replace with your contract ABI

    identity_manager = IdentityManagement(contract_address, abi, provider_url)

    # Example: Register an identity
    account = "0xYourAccountAddress"  # Replace with your Ethereum account address
    private_key = "YourPrivateKey"  # Replace with your private key
    name = "John Doe"
    email = "john.doe@example.com"

    tx_hash = identity_manager.register_identity(account, private_key, name, email)
    print(f "Identity registered! Transaction hash: {tx_hash}")

    # Example: Update an identity
    new_name = "Johnathan Doe"
    new_email = "johnathan.doe@example.com"
    tx_hash = identity_manager.update_identity(account, private_key, new_name, new_email)
    print(f"Identity updated! Transaction hash: {tx_hash}")

    # Example: Retrieve an identity
    identity = identity_manager.get_identity(account)
    print(f"Identity: Name: {identity[0]}, Email: {identity[1]}")

    # Example: Check if identity exists
    exists = identity_manager.identity_exists(account)
    print(f"Identity exists: {exists}")
