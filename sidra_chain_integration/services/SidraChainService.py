class SidraChainService:
    def __init__(self, sidra_chain_model: SidraChainModel) -> None:
        self.sidra_chain_model = sidra_chain_model

    def transfer_funds(self, recipient: str, amount: int) -> None:
        # Call the Sidra chain's transferFunds function using the service's model
        url = f"{self.sidra_chain_model.node_url}/transferFunds"
        headers = {"Authorization": f"Bearer {self.sidra_chain_model.api_key}"}
        data = {"recipient": recipient, "amount": amount}
        response = requests.post(url, headers=headers, json=data)
        # Handle response
        pass
