from pi_network_blockchain_interface import BlockchainInterface


class SmartContractInteraction:
    def __init__(self, blockchain_interface, contract_address, abi):
        self.blockchain_interface = blockchain_interface
        self.contract = self.blockchain_interface.web3.eth.contract(
            address=contract_address, abi=abi
        )

    def call_contract_function(self, function_name, *args):
        return self.contract.functions[function_name](*args).call()

    def send_transaction_to_contract(self, function_name, *args, transaction_value=0):
        return self.contract.functions[function_name](*args).transact(
            {"value": transaction_value}
        )
