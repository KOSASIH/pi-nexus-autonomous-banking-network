# Pi Network Blockchain Explorer
import flask
from pi_network.core.blockchain import Blockchain

app = flask.Flask(__name__)

blockchain = Blockchain()

@app.route("/blocks", methods=["GET"])
def get_blocks():
    # Return list of blocks in blockchain
    return {"blocks": [block.to_dict() for block in blockchain.blocks]}

@app.route("/block/<string:block_hash>", methods=["GET"])
def get_block(block_hash):
    # Return block by hash
    block = blockchain.get_block(block_hash)
    if block:
        return block.to_dict()
    return {"error": "Block not found"}, 404

@app.route("/transactions", methods=["GET"])
def get_transactions():
    # Return list of transactions in blockchain
    return {"transactions": [tx.to_dict() for tx in blockchain.transactions]}

@app.route("/transaction/<string:tx_id>", methods=["GET"])
def get_transaction(tx_id):
    # Return transaction by ID
    tx = blockchain.get_transaction(tx_id)
    if tx:
        return tx.to_dict()
    return {"error": "Transaction not found"}, 404

if __name__ == "__main__":
    app.run(debug=True)
