import json
import requests
import hashlib
import hmac
import base64
import time
import os

from cryptography import Cryptography

class API:
    def __init__(self, host, port, api_key, api_secret):
        self.host = host
        self.port = port
        self.api_key = api_key
        self.api_secret = api_secret

    def _create_signature(self, message):
        signature = hmac.new(
            self.api_secret.encode(),
            message.encode(),
            hashlib.sha256
        ).hexdigest()

        return signature

    def _create_request_headers(self, method, path, data=None):
        timestamp = int(time.time())
        message = f'{timestamp}{method}{path}'

        if data is not None:
            message += json.dumps(data)

        signature = self._create_signature(message)

        headers = {
            'API-Key': self.api_key,
            'API-Timestamp': str(timestamp),
            'API-Signature': signature
        }

        return headers

    def _send_request(self, method, path, data=None):
        url = f'http://{self.host}:{self.port}/{path}'
        headers = self._create_request_headers(method, path, data)

        if method == 'GET':
            response = requests.get(url, headers=headers)
        elif method == 'POST':
            response = requests.post(url, headers=headers, json=data)
        elif method == 'PUT':
            response = requests.put(url, headers=headers, json=data)
        elif method == 'DELETE':
            response = requests.delete(url, headers=headers)
        else:
            raise ValueError(f'Invalid method: {method}')

        if response.status_code != 200:
            raise Exception(f'API request failed with status code: {response.status_code}')

        return response.json()

    def get_balance(self, user_id):
        path = f'users/{user_id}/balance'
        response = self._send_request('GET', path)

        return response['balance']

    def transfer(self, sender_id, receiver_id, amount):
        path = f'transfers'
        data = {
            'sender_id': sender_id,
            'receiver_id': receiver_id,
            'amount': amount
        }
        response = self._send_request('POST', path, data)

        return response['transaction_id']

    def generate_address(self, user_id):
        path = f'users/{user_id}/address'
        response = self._send_request('POST', path)

        return response['address']

    def get_transaction(self, transaction_id):
        path = f'transactions/{transaction_id}'
        response = self._send_request('GET', path)

        return response

    def get_transactions(self, user_id, limit=10, offset=0):
        path = f'users/{user_id}/transactions'
        data = {
            'limit': limit,
            'offset': offset
        }
        response = self._send_request('GET', path, data)

        return response

    def sign_message(self, message):
        cryptography = Cryptography()
        key = cryptography.load_private_key('private_key.pem')
        signature = cryptography.sign_message(message, key)

        return signature

    def verify_message(self, message, signature):
        cryptography = Cryptography()
        public_key = cryptography.load_public_key('public_key.pem')
        is_valid = cryptography.verify_message(message, signature, public_key)

        return is_valid

    def encrypt_message(self, message, recipient_id):
        cryptography = Cryptography()
        public_key = cryptography.load_public_key(f'users/{recipient_id}/public_key.pem')
        ciphertext = cryptography.encrypt_message(message, public_key)

        return ciphertext

    def decrypt_message(self, ciphertext, sender_id):
        cryptography = Cryptography()
        private_key = cryptography.load_private_key(f'users/{sender_id}/private_key.pem')
        plaintext = cryptography.decrypt_message(ciphertext, private_key)

        return plaintext
