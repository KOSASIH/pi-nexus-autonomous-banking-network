# app.py
from flask import Flask, jsonify, request
from flask_sqlalchemy import SQLAlchemy

app = Flask(__name__)
app.config["SQLALCHEMY_DATABASE_URI"] = os.environ["DATABASE_URI"]
db = SQLAlchemy(app)


class Account(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(80), unique=True, nullable=False)
    balance = db.Column(db.Float, default=0.0)


@app.route("/accounts", methods=["GET"])
def get_accounts():
    accounts = Account.query.all()
    return jsonify(
        [
            {"id": account.id, "username": account.username, "balance": account.balance}
            for account in accounts
        ]
    )


if __name__ == "__main__":
    app.run(debug=True)
