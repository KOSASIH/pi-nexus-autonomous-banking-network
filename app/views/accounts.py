from flask import Blueprint, flash, redirect, render_template, request, url_for

from app import db
from app.models import Account, Transaction

accounts = Blueprint("accounts", __name__)


@accounts.route("/accounts")
def accounts_index():
    accounts = Account.query.all()
    return render_template("accounts/index.html", accounts=accounts)


@accounts.route("/accounts/new", methods=["GET", "POST"])
def accounts_new():
    if request.method == "POST":
        username = request.form["username"]
        email = request.form["email"]
        password = request.form["password"]
        balance = request.form["balance"]
        currency_id = request.form["currency_id"]

        account = Account(
            username=username,
            email=email,
            password=password,
            balance=balance,
            currency_id=currency_id,
        )
        db.session.add(account)
        db.session.commit()

        flash("Account created successfully!", "success")
        return redirect(url_for("accounts.accounts_index"))

    currencies = Currency.query.all()
    return render_template("accounts/new.html", currencies=currencies)


@accounts.route("/accounts/<int:account_id>/edit", methods=["GET", "POST"])
def accounts_edit(account_id):
    account = Account.query.get_or_404(account_id)

    if request.method == "POST":
        account.username = request.form["username"]
        account.email = request.form["email"]
        account.password = request.form["password"]
        account.balance = request.form["balance"]
        account.currency_id = request.form["currency_id"]

        db.session.commit()

        flash("Account updated successfully!", "success")
        return redirect(url_for("accounts.accounts_index"))

    currencies = Currency.query.all()
    return render_template("accounts/edit.html", account=account, currencies=currencies)


@accounts.route("/accounts/<int:account_id>/delete", methods=["POST"])
def accounts_delete(account_id):
    account = Account.query.get_or_404(account_id)

    db.session.delete(account)
    db.session.commit()

    flash("Account deleted successfully!", "success")
    return redirect(url_for("accounts.accounts_index"))


@accounts.route("/accounts/<int:account_id>/transactions")
def accounts_transactions(account_id):
    account = Account.query.get_or_404(account_id)
    transactions = account.transactions
    return render_template(
        "accounts/transactions.html", account=account, transactions=transactions
    )


@accounts.route("/accounts/<int:account_id>/transactions/new", methods=["GET", "POST"])
def accounts_transactions_new(account_id):
    account = Account.query.get_or_404(account_id)

    if request.method == "POST":
        receiver_id = request.form["receiver_id"]
        amount = request.form["amount"]
        currency_id = request.form["currency_id"]

        transaction = Transaction(
            sender_id=account_id,
            receiver_id=receiver_id,
            amount=amount,
            currency_id=currency_id,
        )
        db.session.add(transaction)
        db.session.commit()

        flash("Transaction created successfully!", "success")
        return redirect(
            url_for("accounts.accounts_transactions", account_id=account_id)
        )

    receivers = Account.query.all()
    currencies = Currency.query.all()
    return render_template(
        "accounts/transactions_new.html",
        account=account,
        receivers=receivers,
        currencies=currencies,
    )


@accounts.route(
    "/accounts/<int:account_id>/transactions/<int:transaction_id>/delete",
    methods=["POST"],
)
def accounts_transactions_delete(account_id, transaction_id):
    transaction = Transaction.query.get_or_404(transaction_id)

    db.session.delete(transaction)
    db.session.commit()

    flash("Transaction deleted successfully!", "success")
    return redirect(url_for("accounts.accounts_transactions", account_id=account_id))
