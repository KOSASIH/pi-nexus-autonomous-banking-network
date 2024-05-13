import json
import web3

class Ethereum:
    def __init__(self, web3_provider):
        self.web3 = web3.Web3(web3_provider)

    def deploy_contract(self, contract_abi, contract_bin):
        """
        Deploy a Solidity contract to the Ethereum blockchain.
        """
        contract = self.web3.eth.contract(abi=contract_abi, bytecode=contract_bin)
        tx_hash = contract.constructor().transact({'from': self.web3.eth.defaultAccount})
        tx_receipt = self.web3.eth.waitForTransactionReceipt(tx_hash)
        contract_address = tx_receipt['contractAddress']
        contract_instance = self.web3.eth.contract(address=contract_address, abi=contract_abi)
        return contract_instance

    def call_contract_function(self, contract_instance, function_name, *args):
        """
        Call a function of a Solidity contract.
        """
        function = contract_instance.functions[function_name](*args)
        result = function.call()
        return result

    def send_transaction_to_contract(self, contract_instance, function_name, *args, gas=None, gas_price=None):
        """
        Send a transaction to a Solidity contract.
        """
        function = contract_instance.functions[function_name](*args)
        tx = function.buildTransaction({
            'from': self.web3.eth.defaultAccount,
            'gas': gas,
            'gasPrice': gas_price
        })
        signed_tx = self.web3.eth.account.signTransaction(tx, private_key=self.web3.eth.defaultAccount.privateKey.key)
        tx_hash = self.web3.eth.sendRawTransaction(signed_tx.rawTransaction)
        tx_receipt = self.web3.eth.waitForTransactionReceipt(tx_hash)
        return tx_receipt
