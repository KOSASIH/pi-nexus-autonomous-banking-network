# api/new_endpoint.py
from flask import Blueprint, jsonify

new_endpoint = Blueprint('new_endpoint', __name__)

@new_endpoint.route('/new_endpoint', methods=['GET'])
def get_new_endpoint():
    return jsonify({'message': 'New endpoint'})
