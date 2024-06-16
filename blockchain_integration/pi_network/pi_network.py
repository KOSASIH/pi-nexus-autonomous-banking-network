import os
import json
import requests
from pi_network_sdk import PiNetworkSDK

class PiNetwork:
    def __init__(self, config_file='pi_network_config.json'):
        self.config = self.load_config(config_file)
        self.sdk = PiNetworkSDK(self.config['api_key'], self.config['api_secret'])
        self.nodes = {}

    def load_config(self, config_file):
        with open(config_file, 'r') as f:
            return json.load(f)

    def connect(self):
        self.sdk.connect()

    def get_nodes(self):
        response = self.sdk.get_nodes()
        for node in response['nodes']:
            self.nodes[node['id']] = PiNode(node)
        return self.nodes

    def get_node(self, node_id):
        return self.nodes.get(node_id)

    def send_command(self, node_id, command):
        self.sdk.send_command(node_id, command)

    def get_data(self, node_id, data_type):
        return self.sdk.get_data(node_id, data_type)

    def close(self):
        self.sdk.close()

class PiNode:
    def __init__(self, node_data):
        self.id = node_data['id']
        self.name = node_data['name']
        self.status = node_data['status']
        self.data = {}

    def update_status(self, status):
        self.status = status

    def add_data(self, data_type, data):
        self.data[data_type] = data

    def get_data(self, data_type):
        return self.data.get(data_type)
