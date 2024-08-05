import requests
import json
from flask import Flask, request, jsonify
from flask_restful import Api, Resource
from pi_network_node import PINetworkNode

app = Flask(__name__)
api = Api(app)

class PINetworkAPI:
    def __init__(self, node):
        self.node = node
        self.nodes = {}

    def register_node(self, node_id, node_address):
        self.nodes[node_id] = node_address
        return {'message': f'Node {node_id} registered successfully'}

    def unregister_node(self, node_id):
        if node_id in self.nodes:
            del self.nodes[node_id]
            return {'message': f'Node {node_id} unregistered successfully'}
        else:
            return {'message': f'Node {node_id} not found'}, 404

    def get_nodes(self):
        return {'nodes': list(self.nodes.keys())}

    def send_message(self, node_id, message):
        if node_id in self.nodes:
            response = requests.post(f'http://{self.nodes[node_id]}/receive_message', json={'message': message})
            if response.status_code == 200:
                return {'message': f'Message sent to node {node_id} successfully'}
            else:
                return {'message': f'Error sending message to node {node_id}'}, 500
        else:
            return {'message': f'Node {node_id} not found'}, 404

class NodeRegistration(Resource):
    def post(self):
        node_id = request.json['node_id']
        node_address = request.json['node_address']
        return api.register_node(node_id, node_address)

class NodeUnregistration(Resource):
    def post(self):
        node_id = request.json['node_id']
        return api.unregister_node(node_id)

class NodeList(Resource):
    def get(self):
        return api.get_nodes()

class SendMessage(Resource):
    def post(self):
        node_id = request.json['node_id']
        message = request.json['message']
        return api.send_message(node_id, message)

api.add_resource(NodeRegistration, '/register_node')
api.add_resource(NodeUnregistration, '/unregister_node')
api.add_resource(NodeList, '/nodes')
api.add_resource(SendMessage, '/send_message')

if __name__ == '__main__':
    node = PINetworkNode()
    api = PINetworkAPI(node)
    app.run(debug=True)
