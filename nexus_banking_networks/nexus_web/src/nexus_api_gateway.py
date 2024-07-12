import json
import os

from flask import Flask, jsonify, request
from flask_restful import Api, Resource

app = Flask(__name__)
api = Api(app)


class NexusAPIGateway(Resource):

    def post(self):
        data = request.get_json()
        # Process data and perform necessary actions
        return jsonify({"message": "Data processed successfully"})


api.add_resource(NexusAPIGateway, "/api/gateway")

if __name__ == "__main__":
    app.run(debug=True)
