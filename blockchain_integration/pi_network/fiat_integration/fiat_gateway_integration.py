import os
import logging
import requests
from cryptography.fernet import Fernet
from typing import Dict, List
from cachetools import cached, TTLCache
from config import Config

# Configuration
config = Config()

# Logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Error Handling
class FiatGatewayError(Exception):
    pass

class FiatGatewayIntegration:
    def __init__(self):
        self.api_key = config.FIAT_GATEWAY_API_KEY
        self.api_secret = config.FIAT_GATEWAY_API_SECRET
        self.url = config.FIAT_GATEWAY_URL
        self.session = requests.Session()
        self.cache = TTLCache(maxsize=100, ttl=300)  # 5-minute cache

    def _generate_signature(self, payload: Dict) -> str:
        # Generate a secure signature using the API secret
        fernet = Fernet(self.api_secret.encode())
        signature = fernet.encrypt(json.dumps(payload).encode())
        return signature.decode()

    def _make_request(self, method: str, endpoint: str, payload: Dict = {}) -> Dict:
        # Make a request to the fiat gateway API
        headers = {
            'Authorization': f'Bearer {self.api_key}',
            'Content-Type': 'application/json'
        }
        if payload:
            payload['signature'] = self._generate_signature(payload)
            response = self.session.request(method, f'{self.url}/{endpoint}', json=payload, headers=headers)
        else:
            response = self.session.request(method, f'{self.url}/{endpoint}', headers=headers)
        response.raise_for_status()
        return response.json()

    @cached(cache)
    def get_fiat_rates(self) -> List[Dict]:
        # Get the current fiat rates
        endpoint = 'rates'
        response = self._make_request('GET', endpoint)
        return response['rates']

    def swap_pi_coins(self, amount: int, fiat_currency: str) -> Dict:
        # Swap Pi coins to fiat currency
        endpoint = 'swap'
        payload = {
            'amount': amount,
            'fiat_currency': fiat_currency
        }
        response = self._make_request('POST', endpoint, payload)
        return response

    @cached(cache)
    def get_transaction_history(self) -> List[Dict]:
        # Get the transaction history
        endpoint = 'transactions'
        response = self._make_request('GET', endpoint)
        return response['transactions']

    def get_fiat_balance(self, fiat_currency: str) -> Dict:
        # Get the fiat balance
        endpoint = 'balance'
        payload = {
            'fiat_currency': fiat_currency
        }
        response = self._make_request('GET', endpoint, payload)
        return response

    def transfer_fiat(self, amount: int, fiat_currency: str, recipient: str) -> Dict:
        # Transfer fiat to another user
        endpoint = 'transfer'
        payload = {
            'amount': amount,
            'fiat_currency': fiat_currency,
            'recipient': recipient
        }
        response = self._make_request('POST', endpoint, payload)
        return response

if __name__ == '__main__':
    # Testing
    fiat_gateway = FiatGatewayIntegration()
    print(fiat_gateway.get_fiat_rates())
    print(fiat_gateway.swap_pi_coins(100, 'USD'))
    print(fiat_gateway.get_transaction_history())
    print(fiat_gateway.get_fiat_balance('USD'))
    print(fiat_gateway.transfer_fiat(100, 'USD', 'recipient@example.com'))
