import ariadne
from ariadne.constants import PLAYGROUND_HTML
from flask import Flask, request, jsonify

app = Flask(__name__)

type_defs = '''
    type Transaction {
        id: ID!
        from: String!
        to: String!
        value: Int!
        timestamp: Int!
    }

    type Query {
        transactions: [Transaction!]!
    }
'''

query = ariadne.QueryType()

@query.field("transactions")
def resolve_transactions(_, info):
    # Connect to blockchain node
    w3 = Web3(Web3.HTTPProvider('https://mainnet.infura.io/v3/YOUR_PROJECT_ID'))
    # Get latest transactions
    transactions = w3.eth.get_transaction_count()
    # Process and return transactions
    return [{'id': tx['hash'], 'from': tx['from'], 'to': tx['to'], 'value': tx['value'], 'timestamp': tx['timestamp']} for tx in transactions]

schema = ariadne.make_executable_schema(type_defs, query)

@app.route('/graphql', methods=['GET'])
def graphql_playground():
    return PLAYGROUND_HTML, 200

@app.route('/graphql', methods=['POST'])
def graphql_server():
    data = request.get_json()
    success, result = ariadne.graphql_sync(schema, data, context_value=request)
    return jsonify(result), 200 if success else 400

if __name__ == '__main__':
    app.run(debug=True)
