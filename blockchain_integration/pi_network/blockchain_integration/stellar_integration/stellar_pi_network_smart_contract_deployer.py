import stellar_sdk

class StellarPiNetworkSmartContractDeployer:
    def __init__(self, network_passphrase, horizon_url):
        self.network_passphrase = network_passphrase
        self.horizon_url = horizon_url
        self.server = stellar_sdk.Server(horizon_url)

    def deploy_smart_contract(self, contract_code):
        # Deploy a smart contract on the Stellar blockchain
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
