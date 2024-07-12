import flask
from flask_restful import Api, Resource, reqparse

app = flask.Flask(__name__)
api = Api(app)

class Transaction(Resource):
    def post(self):
        parser = reqparse.RequestParser()
        parser.add_argument('amount', type=float, required=True)
        parser.add_argument('account_number', type=str, required=True)
        args = parser.parse_args()
        # Process transaction logic here
        return {'message': 'Transaction successful'}, 201

api.add_resource(Transaction, '/transaction')

if __name__ == '__main__':
    app.run(debug=True)
