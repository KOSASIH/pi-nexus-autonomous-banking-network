from flask import Blueprint, render_template, request, redirect, url_for, flash
from app.models import Currency
from app import db

currencies = Blueprint('currencies', __name__)

@currencies.route('/currencies')
def currencies_index():
    currencies = Currency.query.all()
    return render_template('currencies/index.html', currencies=currencies)

@currencies.route('/currencies/new', methods=['GET', 'POST'])
def currencies_new():
    if request.method == 'POST':
        name = request.form['name']
        symbol = request.form['symbol']
        exchange_rate = request.form['exchange_rate']

        currency = Currency(name=name, symbol=symbol, exchange_rate=exchange_rate)
        db.session.add(currency)
        db.session.commit()

        flash('Currency created successfully!', 'success')
        return redirect(url_for('currencies.currencies_index'))

    return render_template('currencies/new.html')

@currencies.route('/currencies/<int:currency_id>/edit', methods=['GET', 'POST'])
def currencies_edit(currency_id):
    currency = Currency.query.get_or_404(currency_id)

    if request.method == 'POST':
        currency.name = request.form['name']
        currency.symbol = request.form['symbol']
        currency.exchange_rate = request.form['exchange_rate']

        db.session.commit()

        flash('Currency updated successfully!', 'success')
        return redirect(url_for('currencies.currencies_index'))

    return render_template('currencies/edit.html', currency=currency)

@currencies.route('/currencies/<int:currency_id>/delete', methods=['POST'])
def currencies_delete(currency_id):
    currency = Currency.query.get_or_404(currency_id)

    db.session.delete(currency)
    db.session.commit()

    flash('Currency deleted successfully!', 'success')
    return redirect(url_for('currencies.currencies_index'))

@currencies.route('/currencies/<int:currency_id>/exchange_rate', methods=['GET', 'POST'])
def currencies_exchange_rate(currency_id):
    currency = Currency.query.get_or_404(currency_id)

    if request.method == 'POST':
        exchange_rate = request.form['exchange_rate']
        currency.exchange_rate = exchange_rate

        db.session.commit()

        flash('Exchange rate updated successfully!', 'success')
        return redirect(url_for('currencies.currencies_index'))

    return render_template('currencies/exchange_rate.html', currency=currency)
