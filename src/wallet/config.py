class Config:
    def __init__(self):
        self.network = "mainnet"  # or "testnet"
        self.api_url = "https://api.example.com"
        self.default_gas_price = 1000000000  # in wei
        self.supported_tokens = ["ETH", "BTC", "USDT"]  # Example supported tokens

    def get_config(self):
        return {
            "network": self.network,
            "api_url": self.api_url,
            "default_gas_price": self.default_gas_price,
            "supported_tokens": self.supported_tokens
        }
