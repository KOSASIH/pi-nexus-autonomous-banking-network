from typing import Dict, List

from jinja2 import Environment, FileSystemLoader


def render_account_balance(account: Dict) -> str:
    env = Environment(loader=FileSystemLoader("templates"))
    template = env.get_template("account_balance.html")
    return template.render(account=account)


def render_transaction_history(transactions: List[Dict]) -> str:
    env = Environment(loader=FileSystemLoader("templates"))
    template = env.get_template("transaction_history.html")
    return template.render(transactions=transactions)
