from flask import Blueprint

from . import accounts, currencies, transactions

views = Blueprint("views", __name__)

views.register_blueprint(accounts)
views.register_blueprint(transactions)
views.register_blueprint(currencies)
