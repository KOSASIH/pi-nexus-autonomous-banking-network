import requests

class PiNetwork:
    def __init__(self, node_url):
        self.node_url = node_url

    def get_balance(self, address):
        url = f"{self.node_url}/balance/{address}"
        response = requests.get(url)
        return response.json()["balance"]

    def send_transaction(self, sender, receiver, amount):
        url = f"{self.node_url}/transaction"
        data = {"sender": sender, "receiver": receiver, "amount": amount}
        response = requests.post(url, json=data)
        return response.json()["tx_hash"]
