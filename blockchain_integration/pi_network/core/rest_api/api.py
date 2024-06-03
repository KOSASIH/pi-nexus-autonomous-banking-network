# api.py
from flask import Flask, jsonify, request

from .models import Transaction, User

app = Flask(__name__)


@app.route("/users", methods=["GET"])
def get_users():
    # Return a list of users
    pass


@app.route("/transactions", methods=["GET"])
def get_transactions():
    # Return a list of transactions
    pass
