from flask import Blueprint, jsonify
from models import Blockchain

blockchain_blueprint = Blueprint('blockchain', __name__)

@blockchain_blueprint.route('/blockchain', methods=['GET'])
def get_blockchain():
  blockchain = Blockchain()
  return jsonify(blockchain.chain)

@blockchain_blueprint.route('/blockchain/add_block', methods=['POST'])
def add_block():
  blockchain = Blockchain()
  new_block = Block(len(blockchain.chain), blockchain.get_latest_block().hash, int(time.time()), "New Block")
  blockchain.add_block(new_block)
  return jsonify(blockchain.chain)
