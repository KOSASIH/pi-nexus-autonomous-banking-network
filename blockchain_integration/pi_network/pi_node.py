import os
import json
from pi_network_sdk import PiNetworkSDK

class PiNode:
    def __init__(self, node_id, config_file='pi_node_config.json'):
        self.config = self.load_config(config_file)
        self.sdk = PiNetworkSDK(self.config['api_key'], self.config['api_secret'])
        self.node_id = node_id
        self.status = 'offline'
        self.data = {}

    def load_config(self, config_file):
        with open(config_file, 'r') as f:
            return json.load(f)

    def connect(self):
        self.sdk.connect()

    def update_status(self, status):
        self.status = status
        self.sdk.update_node_status(self.node_id, status)

    def send_data(self, data_type, data):
        self.sdk.send_data(self.node_id, data_type, data)

    def get_data(self, data_type):
        return self.sdk.get_data(self.node_id, data_type)

    def close(self):
        self.sdk.close()
