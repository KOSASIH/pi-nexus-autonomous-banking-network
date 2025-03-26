import json

import requests


class PiNetworkBlockchainExplorer:
    def __init__(self):
        self.api_url = "http://localhost:8080/json_rpc"

    def get_blockchain_info(self):
        params = {
            "jsonrpc": "2.0",
            "method": "getblockchaininfo",
            "params": [],
            "id": 1,
        }
        response = requests.post(self.api_url, json=params)
        return json.loads(response.text)

    def get_block_by_height(self, height):
        params = {
            "jsonrpc": "2.0",
            "method": "getblockbyheight",
            "params": [height],
            "id": 1,
        }
        response = requests.post(self.api_url, json=params)
        return json.loads(response.text)

    def get_transaction_by_hash(self, tx_hash):
        params = {
            "jsonrpc": "2.0",
            "method": "gettransaction",
            "params": [tx_hash],
            "id": 1,
        }
        response = requests.post(self.api_url, json=params)
        return json.loads(response.text)
