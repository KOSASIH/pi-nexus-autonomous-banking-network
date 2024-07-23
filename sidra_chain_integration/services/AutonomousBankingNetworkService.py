class AutonomousBankingNetworkService:
    def __init__(
        self, autonomous_banking_network_model: AutonomousBankingNetworkModel
    ) -> None:
        self.autonomous_banking_network_model = autonomous_banking_network_model

    def transfer_funds(self, recipient: str, amount: int) -> None:
        # Call the Autonomous Banking Network's transferFunds function using the service's model
        self.autonomous_banking_network_model.transfer_funds(recipient, amount)

    def get_account_balance(self, account_address: str) -> int:
        # Query the Autonomous Banking Network's account balance using the service's model
        return self.autonomous_banking_network_model.get_account_balance(
            account_address
        )
