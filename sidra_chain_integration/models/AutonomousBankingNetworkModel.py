@dataclass
class AutonomousBankingNetworkModel:
    node_url: str
    api_key: str
    contract_address: str

    def transfer_funds(self, recipient: str, amount: int) -> None:
        # Call the Autonomous Banking Network's transferFunds function using the contract address and API key
        url = f"{self.node_url}/transferFunds"
        headers = {"Authorization": f"Bearer {self.api_key}"}
        data = {"recipient": recipient, "amount": amount}
        response = requests.post(url, headers=headers, json=data)
        # Handle response
        pass

    def get_account_balance(self, account_address: str) -> int:
        # Query the Autonomous Banking Network's account balance using the contract address and API key
        url = f"{self.node_url}/getAccountBalance"
        headers = {"Authorization": f"Bearer {self.api_key}"}
        params = {"accountAddress": account_address}
        response = requests.get(url, headers=headers, params=params)
        # Handle response
        pass
