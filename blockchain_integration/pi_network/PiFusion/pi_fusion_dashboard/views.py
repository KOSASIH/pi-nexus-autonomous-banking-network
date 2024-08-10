from flask import render_template
from flask_restful import Resource
from models import Node, Transaction

class NodeView(Resource):
    def get(self, node_id):
        node = Node.query.get(node_id)
        return render_template('node.html', node=node)

class TransactionView(Resource):
    def get(self, transaction_id):
        transaction = Transaction.query.get(transaction_id)
        return render_template('transaction.html', transaction=transaction)
