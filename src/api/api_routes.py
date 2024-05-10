from api_models import *
from flask_restful import Resource, reqparse

api_parser = reqparse.RequestParser()
api_parser.add_argument("api_key", required=True, help="API key is required")


class HealthCheckResource(Resource):
    def get(self):
        return {"status": "ok"}


class TransactionResource(Resource):
    def post(self):
        args = api_parser.parse_args()
        api_key = args["api_key"]

        if not validate_api_key(api_key):
            return {"error": "Invalid API key"}, 401

        data = request.get_json()
        transaction = Transaction(data["sender"], data["receiver"], data["amount"])
        transaction_pool.add_transaction(transaction)

        return {"message": "Transaction added to pool"}, 201


api.add_resource(HealthCheckResource, "/")
api.add_resource(TransactionResource, "/transaction")
