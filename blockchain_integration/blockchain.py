# blockchain_integration/blockchain.py
import web3

class Blockchain:
    def __init__(self, provider_url: str):
        self.web3 = web3.Web3(web3.providers.Web3Provider(provider_url))

    def get_balance(self, address: str) -> int:
        try:
            balance = self.web3.eth.get_balance(address)
            return balance
        except web3.exceptions.ContractLogicError as e:
            # Handle error and retry
            print(f"Error: {e}")
            return None
