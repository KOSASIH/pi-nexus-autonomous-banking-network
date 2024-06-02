import json
import os

from flask import Flask, jsonify, request

from .data_loader import DataLoader
from .models import ModelFactory
from .portfolio_analysis import PortfolioAnalyzer
from .security import SecurityManager
from .utils import load_config, save_config

app = Flask(__name__)
config = load_config("config.json")
data_loader = DataLoader(config["data_path"])
analyzer = PortfolioAnalyzer(data_loader)


@app.route("/login", methods=["POST"])
def login():
    user_id = request.form["user_id"]
    password = request.form["password"]
    security_manager = SecurityManager(user_id, password)
    response = {
        "user_id": user_id,
        "key": security_manager.key.decode(),
        "salt": security_manager.salt,
    }
    return jsonify(response)


@app.route("/analyze", methods=["POST"])
def analyze():
    encrypted_data = request.form["data"]
    user_id = request.form["user_id"]
    password = request.form["password"]
    security_manager = SecurityManager(user_id, password)
    decrypted_data = security_manager.decrypt_data(encrypted_data)
    data = json.loads(decrypted_data)
    result = analyzer.analyze_portfolio(data)
    response = {"result": result}
    return jsonify(response)


if __name__ == "__main__":
    app.run(port=config["port"])
