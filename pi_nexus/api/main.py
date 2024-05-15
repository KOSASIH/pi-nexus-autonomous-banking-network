# pi_nexus/api/main.py
from flask import Flask, jsonify
from flask_restful import Api, Resource
from pi_nexus.services import banking_service

app = Flask(__name__)
api = Api(app)

class BankingResource(Resource):
    def get(self):
        banking_data = banking_service.get_banking_data()
        return jsonify(banking_data)

api.add_resource(BankingResource, '/banking')

if __name__ == '__main__':
    app.run(debug=True)
