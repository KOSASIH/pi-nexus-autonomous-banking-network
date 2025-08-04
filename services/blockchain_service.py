# services/blockchain_service.py
import web3


class BlockchainService:
    def __init__(self, contract_address, contract_abi):
        self.web3 = web3.Web3(web3.HTTPProvider("http://localhost:8545"))
        self.contract = self.web3.eth.contract(
            address=contract_address, abi=contract_abi
        )

    def create_transaction(self, from_address, to_address, amount):
        nonce = self.web3.eth.getTransactionCount(from_address)
        transaction = {
            "nonce": nonce,
            "to": to_address,
            "value": amount,
            "gas": 21000,
            "gasPrice": self.web3.eth.gasPrice,
        }
        signed_transaction = self.web3.eth.account.signTransaction(
            transaction, private_key
        )
        tx_hash = self.web3.eth.sendRawTransaction(signed_transaction.rawTransaction)
        return tx_hash
