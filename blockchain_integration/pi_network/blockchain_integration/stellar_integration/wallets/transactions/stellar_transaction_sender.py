import requests

class StellarTransactionSender:
    def __init__(self, horizon_url):
        self.horizon_url = horizon_url

    def send_transaction(self, signed_envelope):
        response = requests.post(self.horizon_url + '/transactions', data=signed_envelope.to_xdr())
        return response.json()
