from flask import Flask, jsonify
from flask_restful import Api, Resource

app = Flask(__name__)
api = Api(app)


class SidraChainIntegration(Resource):
    def get(self):
        return jsonify({"message": "Sidra Chain Integration Service"})


api.add_resource(SidraChainIntegration, "/sidra-chain-integration")

if __name__ == "__main__":
    app.run(debug=True, port=8080)
