import requests

class OracleIntegration:
    def __init__(self, oracle_url):
        self.oracle_url = oracle_url

    def get_price(self, asset):
        response = requests.get(f"{self.oracle_url}/price/{asset}")
        return response.json()

# Example usage
if __name__ == "__main__":
    oracle = OracleIntegration('https://oracle-url/api')
    price = oracle.get_price('ETH')
    print(f"Current ETH Price: {price['price']}")
