from typing import Dict

from jinja2 import Environment, FileSystemLoader


def render_transaction_list(transactions: List[Dict]) -> str:
    env = Environment(loader=FileSystemLoader("templates"))
    template = env.get_template("transaction_list.html")
    return template.render(transactions=transactions)


def render_transaction_details(transaction: Dict) -> str:
    env = Environment(loader=FileSystemLoader("templates"))
    template = env.get_template("transaction_details.html")
    return template.render(transaction=transaction)
