@dataclass
class AutonomousBankingNetworkModel:
    node_url: str
    api_key: str
    contract_address: str

    def transfer_funds(self, recipient: str, amount: int) -> None:
        # Call the Autonomous Banking Network's transferFunds function using the contract address and API key
        pass

    def get_account_balance(self, account_address: str) -> int:
        # Query the Autonomous Banking Network's account balance using the contract address and API key
        pass
