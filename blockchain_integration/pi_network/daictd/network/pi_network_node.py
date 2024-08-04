import requests
import json
from flask import Flask, request, jsonify

class PINetworkNode:
    def __init__(self):
        self.app = Flask(__name__)
        self.node_id = None
        self.node_address = None

    def register(self, node_id, node_address):
        self.node_id = node_id
        self.node_address = node_address
        return {'message': f'Node {node_id} registered successfully'}

    def unregister(self):
        self.node_id = None
        self.node_address = None
        return {'message': 'Node unregistered successfully'}

    def receive_message(self, message):
        print(f'Received message: {message}')
        return {'message': 'Message received successfully'}

    def send_message(self, node_id, message):
        response = requests.post(f'http://{node_id}/receive_message', json={'message': message})
        if response.status_code == 200:
            return {'message': f'Message sent to node {node_id} successfully'}
        else:
            return {'message': f'Error sending message to node {node_id}'}, 500

class ReceiveMessage(Resource):
    def post(self):
        message = request.json['message']
        return node.receive_message(message)

node = PINetworkNode()
node.app.add_resource(ReceiveMessage, '/receive_message')

if __name__ == '__main__':
    node.app.run(debug=True)
