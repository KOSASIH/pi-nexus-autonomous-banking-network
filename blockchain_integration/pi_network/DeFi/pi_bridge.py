# pi_bridge.py

import web3
from web3.contract import Contract

class PIBridge:
    def __init__(self, web3: web3.Web3, contract_address: str):
        self.web3 = web3
        self.contract_address = contract_address
        self.contract = self.web3.eth.contract(address=self.contract_address, abi=self.get_abi())

    def get_abi(self) -> list:
        # Load the PI Bridge ABI from a file or database
        with open('pi_bridge.abi', 'r') as f:
            return json.load(f)

    def transfer(self, asset: str, amount: int, to_chain: str, to_address: str) -> bool:
        # Transfer an asset from one blockchain network to another
        tx_hash = self.contract.functions.transfer(asset, amount, to_chain, to_address).transact({'from': self.web3.eth.accounts[0]})
        return self.web3.eth.waitForTransactionReceipt(tx_hash).status == 1

    def get_transfer_status(self, transfer_id: int) -> str:
        # Get the status of a transfer
        return self.contract.functions.getTransferStatus(transfer_id).call()
