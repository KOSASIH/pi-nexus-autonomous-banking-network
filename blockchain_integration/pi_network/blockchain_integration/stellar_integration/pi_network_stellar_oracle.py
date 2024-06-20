import pi_network
import stellar_sdk

class PiNetworkStellarOracle:
    def __init__(self, pi_network, stellar_client):
        self.pi_network = pi_network
        self.stellar_client = stellar_client

    def get_pi_network_data(self):
        # Retrieve real-time data from the Pi Network
        data = self.pi_network.get_data()
        return data

    def send_data_to_stellar(self, data):
        # Send data to the Stellar blockchain
        transaction = stellar_sdk.TransactionBuilder(
            source_account=self.stellar_client.get_account_id(),
            network_passphrase=self.stellar_client.network_passphrase,
            base_fee=100  # 0.01 XLM
        ).append_data_op(
            data=data
        ).build()
        transaction.sign(self.stellar_client.get_account_id())
        response = self.stellar_client.submit_transaction(transaction)
        return response
