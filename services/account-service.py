from typing import Any, Dict

from flask import Flask, jsonify, request

app = Flask(__name__)

accounts: Dict[int, float] = {}


@app.route("/accounts", methods=["POST"])
def create_account() -> Dict[str, Any]:
    """Creates a new account with a unique account number.

    Returns:
        A JSON object containing the account number and initial balance.
    """
    account_number = len(accounts) + 1
    balance = request.json.get("balance", 0.0)
    accounts[account_number] = balance
    return {"account_number": account_number, "balance": balance}


@app.route("/accounts/<int:account_number>", methods=["GET"])
def get_account(account_number: int) -> Dict[str, Any]:
    """Returns the balance of the specified account.

    Returns:
        A JSON object containing the account balance.
    """
    balance = accounts.get(account_number, None)
    if balance is None:
        return {"error": "Account not found"}, 404
    return {"balance": balance}


@app.route("/accounts/<int:account_number>", methods=["PUT"])
def update_account(account_number: int) -> Dict[str, Any]:
    """Updates the balance of the specified account.

    Returns:
        A JSON object containing the updated account balance.
    """
    balance = request.json.get("balance", None)
    if balance is None:
        return {"error": "Invalid balance"}, 400
    accounts[account_number] = balance
    return {"balance": balance}


if __name__ == "__main__":
    app.run(debug=True)
