import hashlib
from stellar_sdk import Server, TransactionBuilder, Network

class StellarSmartContract:
    def __init__(self, network_passphrase, horizon_url, contract_code):
        self.network_passphrase = network_passphrase
        self.horizon_url = horizon_url
        self.server = Server(horizon_url)
        self.contract_code = contract_code

    def deploy_contract(self, source_keypair):
        transaction = TransactionBuilder(
            source_account=source_keypair.public_key,
            network_passphrase=self.network_passphrase,
            base_fee=100
        ).append_contract_deploy_op(
            contract_code=self.contract_code,
            source_account=source_keypair.public_key
        ).build()
        self.server.submit_transaction(transaction)

    def execute_contract(self, source_keypair, function_name, arguments):
        transaction = TransactionBuilder(
            source_account=source_keypair.public_key,
            network_passphrase=self.network_passphrase,
            base_fee=100
        ).append_contract_execute_op(
            contract_id=self.contract_code,
            function_name=function_name,
            arguments=arguments,
            source_account=source_keypair.public_key
        ).build()
        self.server.submit_transaction(transaction)

    def get_contract_state(self, contract_id):
        contract_state = self.server.contracts().contract_id(contract_id).call()
        return contract_state
