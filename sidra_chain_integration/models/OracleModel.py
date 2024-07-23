@dataclass
class OracleModel:
    node_url: str
    api_key: str
    contract_address: str

    def update_price(self, asset: str, price: int) -> None:
        # Call the Oracle's updatePrice function using the contract address and API key
        url = f"{self.node_url}/updatePrice"
        headers = {"Authorization": f"Bearer {self.api_key}"}
        data = {"asset": asset, "price": price}
        response = requests.post(url, headers=headers, json=data)
        # Handle response
        pass

    def get_price(self, asset: str) -> int:
        # Query the Oracle's getPrice function using the contract address and API key
        url = f"{self.node_url}/getPrice"
        headers = {"Authorization": f"Bearer {self.api_key}"}
        params = {"asset": asset}
        response = requests.get(url, headers=headers, params=params)
        # Handle response
        pass
