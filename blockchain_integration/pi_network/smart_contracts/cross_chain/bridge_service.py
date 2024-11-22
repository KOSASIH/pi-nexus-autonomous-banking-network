import json
from web3 import Web3

class CrossChainBridgeService:
    def __init__(self, contract_address, abi, provider_url):
        self.web3 = Web3(Web3.HTTPProvider(provider_url))
        self.contract = self.web3.eth.contract(address=contract_address, abi=abi)

    def initiate_transfer(self, account, private_key, amount, target_chain, target_address):
        nonce = self.web3.eth.getTransactionCount(account)
        tx = self.contract.functions.initiateTransfer(amount, target_chain, target_address).buildTransaction({
            'chainId': 1,  # Mainnet ID, change as needed
            'gas': 2000000,
            'gasPrice': self.web3.toWei('50', 'gwei'),
            'nonce': nonce,
        })
        signed_tx = self.web3.eth.account.signTransaction(tx, private_key)
        tx_hash = self.web3.eth.sendRawTransaction(signed_tx.rawTransaction)
        return self.web3.toHex(tx_hash)

    def complete_transfer(self, transfer_id, account, private_key):
        nonce = self.web3.eth.getTransactionCount(account)
        tx = self.contract.functions.completeTransfer(transfer_id).buildTransaction({
            'chainId': 1,
            'gas': 2000000,
            'gasPrice': self.web3.toWei('50', 'gwei'),
            'nonce': nonce,
        })
        signed_tx = self.web3.eth.account.signTransaction(tx, private_key)
        tx_hash = self.web3.eth.sendRawTransaction(signed_tx.rawTransaction)
        return self.web3.toHex(tx_hash)

    def get_transfer_details(self, transfer_id):
        return self.contract.functions.getTransferDetails(transfer_id).call()

if __name__ == "__main__":
    # Example usage
    provider_url = "https://your.ethereum.node"  # Replace with your Ethereum node URL
    contract_address = "0xYourContractAddress"  # Replace with your deployed contract address
    abi = json.loads('[{"inputs":[{"internalType":"uint256","name":"_amount","type":"uint256"},{"internalType":"string","name":"_targetChain","type":"string"},{"internalType":"address","name":"_targetAddress","type":"address"}],"name":"initiateTransfer","outputs":[],"stateMutability":"nonpayable","type":" function"},{"inputs":[{"internalType":"bytes32","name":"_transferId","type":"bytes32"}],"name":"completeTransfer","outputs":[],"stateMutability":"nonpayable","type":"function"},{"inputs":[{"internalType":"bytes32","name":"_transferId","type":"bytes32"}],"name":"getTransferDetails","outputs":[{"internalType":"address","name":"","type":"address"},{"internalType":"uint256","name":"","type":"uint256"},{"internalType":"string","name":"","type":"string"},{"internalType":"address","name":"","type":"address"},{"internalType":"bool","name":"","type":"bool"}],"stateMutability":"view","type":"function"}]')  # Replace with your contract ABI

    bridge_service = CrossChainBridgeService(contract_address, abi, provider_url)

    # Example of initiating a transfer
    # tx_hash = bridge_service.initiate_transfer("0xYourAccount", "YourPrivateKey", 1000000000000000000, "Binance Smart Chain", "0xTargetAddress")
    # print(f"Transfer initiated, tx hash: {tx_hash}")

    # Example of completing a transfer
    # transfer_id = "0xYourTransferId"  # Replace with the actual transfer ID
    # tx_hash = bridge_service.complete_transfer(transfer_id, "0xYourAccount", "YourPrivateKey")
    # print(f"Transfer completed, tx hash: {tx_hash}")

    # Example of getting transfer details
    # details = bridge_service.get_transfer_details("0xYourTransferId")
    # print(f"Transfer details: {details}")
