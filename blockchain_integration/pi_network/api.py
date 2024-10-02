import requests


class PiNetworkAPI:
    def __init__(self, api_key, api_secret):
        self.api_key = api_key
        self.api_secret = api_secret

    def init(self):
        # Initialize the API client
        pass

    def get_balance(self, wallet_address):
        # Make the API request
        response = requests.get(
            f"https://api.pi.network/v1/balance?address={wallet_address}&api_key={self.api_key}&api_secret={self.api_secret}"
        )

        # Return the response
        return response
