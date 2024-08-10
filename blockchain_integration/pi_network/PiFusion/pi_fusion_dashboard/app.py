from flask import Flask, render_template, request, jsonify
from flask_sqlalchemy import SQLAlchemy
from flask_marshmallow import Marshmallow
from flask_restful import Api, Resource
from flask_cors import CORS
from models import db, Node, Transaction
from views import NodeView, TransactionView

app = Flask(__name__)
app.config['SQLALCHEMY_DATABASE_URI'] = 'postgresql://user:password@host:port/dbname'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

db.init_app(app)
ma = Marshmallow(app)
api = Api(app)
cors = CORS(app, resources={r"/api/*": {"origins": "*"}})

class NodeResource(Resource):
    def get(self, node_id):
        node = Node.query.get(node_id)
        return node_schema.dump(node)

    def put(self, node_id):
        node = Node.query.get(node_id)
        node.name = request.json['name']
        node.description = request.json['description']
        db.session.commit()
        return node_schema.dump(node)

class TransactionResource(Resource):
    def get(self, transaction_id):
        transaction = Transaction.query.get(transaction_id)
        return transaction_schema.dump(transaction)

    def post(self):
        transaction = Transaction(
            type=request.json['type'],
            amount=request.json['amount'],
            timestamp=request.json['timestamp'],
            node_id=request.json['node_id']
        )
        db.session.add(transaction)
        db.session.commit()
        return transaction_schema.dump(transaction)

api.add_resource(NodeResource, '/api/nodes/<int:node_id>')
api.add_resource(TransactionResource, '/api/transactions/<int:transaction_id>')
api.add_resource(TransactionResource, '/api/transactions', endpoint='transactions')

@app.route('/')
def index():
    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)
