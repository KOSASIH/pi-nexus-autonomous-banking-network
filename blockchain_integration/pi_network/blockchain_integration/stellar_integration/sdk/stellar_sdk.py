import requests

class StellarSDK:
    def __init__(self, horizon_url):
        self.horizon_url = horizon_url

    def get_account(self, account_id):
        response = requests.get(self.horizon_url + '/accounts/' + account_id)
        return response.json()

    def get_transaction(self, transaction_id):
        response =requests.get(self.horizon_url + '/transactions/' + transaction_id)
        return response.json()

    def submit_transaction(self, signed_envelope):
        response = requests.post(self.horizon_url + '/transactions', data=signed_envelope.to_xdr())
        return response.json()
