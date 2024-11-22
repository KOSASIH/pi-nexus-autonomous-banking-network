import json
from web3 import Web3

class TokenizationService:
    def __init__(self, contract_address, abi, provider_url):
        self.web3 = Web3(Web3.HTTPProvider(provider_url))
        self.contract = self.web3.eth.contract(address=contract_address, abi=abi)

    def create_asset(self, account, private_key, name, description, value):
        nonce = self.web3.eth.getTransactionCount(account)
        tx = self.contract.functions.createAsset(name, description, value).buildTransaction({
            'chainId': 1,  # Mainnet ID, change as needed
            'gas': 2000000,
            'gasPrice': self.web3.toWei('50', 'gwei'),
            'nonce': nonce,
        })
        signed_tx = self.web3.eth.account.signTransaction(tx, private_key)
        tx_hash = self.web3.eth.sendRawTransaction(signed_tx.rawTransaction)
        return self.web3.toHex(tx_hash)

    def mint_tokens(self, account, private_key, asset_id, to, amount):
        nonce = self.web3.eth.getTransactionCount(account)
        tx = self.contract.functions.mintTokens(asset_id, to, amount).buildTransaction({
            'chainId': 1,
            'gas': 2000000,
            'gasPrice': self.web3.toWei('50', 'gwei'),
            'nonce': nonce,
        })
        signed_tx = self.web3.eth.account.signTransaction(tx, private_key)
        tx_hash = self.web3.eth.sendRawTransaction(signed_tx.rawTransaction)
        return self.web3.toHex(tx_hash)

    def burn_tokens(self, account, private_key, asset_id, amount):
        nonce = self.web3.eth.getTransactionCount(account)
        tx = self.contract.functions.burnTokens(asset_id, amount).buildTransaction({
            'chainId': 1,
            'gas': 2000000,
            'gas ```python
            'gasPrice': self.web3.toWei('50', 'gwei'),
            'nonce': nonce,
        })
        signed_tx = self.web3.eth.account.signTransaction(tx, private_key)
        tx_hash = self.web3.eth.sendRawTransaction(signed_tx.rawTransaction)
        return self.web3.toHex(tx_hash)

    def get_asset_details(self, asset_id):
        return self.contract.functions.getAssetDetails(asset_id).call()

if __name__ == "__main__":
    # Example usage
    provider_url = "https://your.ethereum.node"  # Replace with your Ethereum node URL
    contract_address = "0xYourContractAddress"  # Replace with your deployed contract address
    abi = json.loads('[{"inputs":[{"internalType":"string","name":"_name","type":"string"},{"internalType":"string","name":"_description","type":"string"},{"internalType":"uint256","name":"_value","type":"uint256"}],"name":"createAsset","outputs":[],"stateMutability":"nonpayable","type":"function"},{"inputs":[{"internalType":"uint256","name":"_assetId","type":"uint256"},{"internalType":"address","name":"_to","type":"address"},{"internalType":"uint256","name":"_amount","type":"uint256"}],"name":"mintTokens","outputs":[],"stateMutability":"nonpayable","type":"function"},{"inputs":[{"internalType":"uint256","name":"_assetId","type":"uint256"},{"internalType":"uint256","name":"_amount","type":"uint256"}],"name":"burnTokens","outputs":[],"stateMutability":"nonpayable","type":"function"},{"inputs":[{"internalType":"uint256","name":"_assetId","type":"uint256"}],"name":"getAssetDetails","outputs":[{"internalType":"string","name":"","type":"string"},{"internalType":"string","name":"","type":"string"},{"internalType":"uint256","name":"","type":"uint256"},{"internalType":"bool","name":"","type":"bool"}],"stateMutability":"view","type":"function"}]')  # Replace with your contract ABI

    tokenization_service = TokenizationService(contract_address, abi, provider_url)

    # Example of creating an asset
    # tx_hash = tokenization_service.create_asset("0xYourAccount", "YourPrivateKey", "Real Estate", "A beautiful property", 1000000000000000000)
    # print(f"Asset created, tx hash: {tx_hash}")

    # Example of minting tokens
    # tx_hash = tokenization_service.mint_tokens("0xYourAccount", "YourPrivateKey", 0, "0xRecipientAddress", 100)
    # print(f"Tokens minted, tx hash: {tx_hash}")

    # Example of burning tokens
    # tx_hash = tokenization_service.burn_tokens("0xYourAccount", "YourPrivateKey", 0, 50)
    # print(f"Tokens burned, tx hash: {tx_hash}")

    # Example of getting asset details
    # details = tokenization_service.get_asset_details(0)
    # print(f"Asset details: {details}")
