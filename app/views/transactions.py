from flask import Blueprint, render_template, request, redirect, url_for, flash
from app.models import Account, Transaction
from app import db

transactions = Blueprint('transactions', __name__)

@transactions.route('/transactions')
def transactions_index():
    transactions = Transaction.query.all()
    return render_template('transactions/index.html', transactions=transactions)

@transactions.route('/transactions/new', methods=['GET', 'POST'])
def transactions_new():
    if request.method == 'POST':
        sender_id = request.form['sender_id']
        receiver_id = request.form['receiver_id']
        amount = request.form['amount']
        currency_id = request.form['currency_id']

        sender = Account.query.get_or_404(sender_id)
        receiver = Account.query.get_or_404(receiver_id)
        currency = Currency.query.get_or_404(currency_id)

        if sender.balance < amount:
            flash('Insufficient balance!', 'danger')
            return redirect(url_for('transactions.transactions_new'))

        transaction = Transaction(sender_id=sender_id, receiver_id=receiver_id, amount=amount, currency_id=currency_id)
        db.session.add(transaction)
        db.session.commit()

        sender.balance -= amount
        receiver.balance += amount
        db.session.commit()

        flash('Transaction created successfully!', 'success')
        return redirect(url_for('transactions.transactions_index'))

    accounts = Account.query.all()
    currencies = Currency.query.all()
    return render_template('transactions/new.html', accounts=accounts, currencies=currencies)

@transactions.route('/transactions/<int:transaction_id>/edit', methods=['GET', 'POST'])
def transactions_edit(transaction_id):
    transaction = Transaction.query.get_or_404(transaction_id)

    if request.method == 'POST':
        transaction.amount = request.form['amount']
        transaction.currency_id = request.form['currency_id']

        db.session.commit()

        flash('Transaction updated successfully!', 'success')
        return redirect(url_for('transactions.transactions_index'))

    currencies = Currency.query.all()
    return render_template('transactions/edit.html', transaction=transaction, currencies=currencies)

@transactions.route('/transactions/<int:transaction_id>/delete', methods=['POST'])
def transactions_delete(transaction_id):
    transaction = Transaction.query.get_or_404(transaction_id)

    sender = Account.query.get_or_404(transaction.sender_id)
    receiver = Account.query.get_or_404(transaction.receiver_id)

    sender.balance += transaction.amount
    receiver.balance -= transaction.amount
    db.session.delete(transaction)
    db.session.commit()

    flash('Transaction deleted successfully!', 'success')
    return redirect(url_for('transactions.transactions_index'))
