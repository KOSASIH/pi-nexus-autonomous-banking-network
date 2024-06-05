app = Flask(__name__)


@app.route("/banks", methods=["GET"])
def get_banks():
    banks = []
    for bank in nexus_network.get_nodes():
        if bank.node_type == "bank":
            banks.append({"bank_id": bank.node_id, "name": bank.attributes["name"]})
    return jsonify(banks)


@app.route("/banks/<bank_id>/accounts", methods=["GET"])
def get_accounts(bank_id):
    bank = nexus_network.get_node(bank_id)
    if bank:
        accounts = []
        for account in bank.accounts.values():
            accounts.append(
                {
                    "account_id": account.account_id,
                    "account_type": account.account_type,
                    "balance": account.balance,
                }
            )
        return jsonify(accounts)
    return jsonify({"error": "Bank not found"}), 404


@app.route("/banks/<bank_id>/accounts/<account_id>/transactions", methods=["GET"])
def get_transactions(bank_id, account_id):
    bank = nexus_network.get_node(bank_id)
    if bank:
        account = bank.get_account(account_id)
        if account:
            transactions = []
            for transaction in account.transactions:
                transactions.append(
                    {
                        "transaction_id": transaction.transaction_id,
                        "amount": transaction.amount,
                        "attributes": transaction.attributes,
                    }
                )
            return jsonify(transactions)
        return jsonify({"error": "Account not found"}), 404
    return jsonify({"error": "Bank not found"}), 404
