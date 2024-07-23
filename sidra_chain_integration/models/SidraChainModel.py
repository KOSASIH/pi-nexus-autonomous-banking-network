@dataclass
class SidraChainModel:
    node_url: str = "https://sidrachain.com"
    api_key: str
    contract_address: str

    def transfer_funds(self, recipient: str, amount: int) -> None:
        # Call the Sidra chain's transferFunds function using the contract address and API key
        pass
