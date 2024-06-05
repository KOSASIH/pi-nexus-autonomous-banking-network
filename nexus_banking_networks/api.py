from flask import Flask, request
from flask_restful import Api, Resource
from nexus_banking_network import NexusAccount, NexusLedger, NexusTransaction

app = Flask(__name__)
api = Api(app)

ledger = NexusLedger()


class AccountResource(Resource):

    def get(self, account_id):
        account = next(
            (acc for acc in ledger.accounts if acc.account_id == account_id), None
        )
        if account:
            return account.__dict__
        else:
            return {"error": "Account not found"}, 404

    def post(self, account_id):
        # Implement creating a new account
        pass


class TransactionResource(Resource):

    def post(self):
        # Implement creating a new transaction
        pass


api.add_resource(AccountResource, "/accounts/<string:account_id>")
api.add_resource(TransactionResource, "/transactions")

if __name__ == "__main__":
    app.run(debug=True)
