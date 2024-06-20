import stellar_sdk

class StellarSmartContract:
    def __init__(self, network_passphrase, horizon_url):
        self.network_passphrase = network_passphrase
        self.horizon_url = horizon_url
        self.server = stellar_sdk.Server(horizon_url)

    def create_contract(self, contract_code):
        # Create a new Stellar smart contract
        transaction = stellar_sdk.TransactionBuilder(
            source_account=self.server.accounts().get_account_id(),
            network_passphrase=self.network_passphrase,
            base_fee=100  # 0.01 XLM
        ).append_contract_op(
            contract_code=contract_code,
            contract_id="pi-network-contract"
        ).build()
        transaction.sign(self.server.accounts().get_account_id())
        response = self.server.submit_transaction(transaction)
        return response

    def execute_contract(self, contract_id, function_name, arguments):
        # Execute a function on the Stellar smart contract
        transaction = stellar_sdk.TransactionBuilder(
            source_account=self.server.accounts().get_account_id(),
            network_passphrase=self.network_passphrase,
            base_fee=100  # 0.01 XLM
        ).append_contract_exec_op(
            contract_id=contract_id,
            function_name=function_name,
            arguments=arguments
        ).build()
        transaction.sign(self.server.accounts().get_account_id())
        response = self.server.submit_transaction(transaction)
        return response
