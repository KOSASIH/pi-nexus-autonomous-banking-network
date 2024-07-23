class SidraChainBridgeService:
    def __init__(self, sidra_chain_bridge_model: SidraChainBridgeModel) -> None:
        self.sidra_chain_bridge_model = sidra_chain_bridge_model

    def transfer_funds(self, recipient: str, amount: int) -> None:
        # Call the Sidra Chain Bridge's transferFunds function using the service's model
        self.sidra_chain_bridge_model.transfer_funds(recipient, amount)

    def get_account_balance(self, account_address: str) -> int:
        # Query the Sidra Chain Bridge's account balance using the service's model
        return self.sidra_chain_bridge_model.get_account_balance(account_address)
