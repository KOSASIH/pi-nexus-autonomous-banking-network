# main.py
import os

import bank_of_america
import capital_one
import chase_bank

api_key = os.getenv("API_KEY")

chase = chase_bank.ChaseBank(api_key)
boa = bank_of_america.BankOfAmerica(api_key)
capitalone = capital_one.CapitalOne(api_key)

chase_accounts = chase.get_accounts()
boa_accounts = boa.get_accounts()
capitalone_accounts = capitalone.get_accounts()

chase_transactions = chase.get_transactions(chase_accounts[0]["id"])
boa_transactions = boa.get_transactions(boa_accounts[0]["id"])
capitalone_transactions = capitalone.get_transactions(capitalone_accounts[0]["id"])

transfer_result = chase.transfer_funds(
    chase_accounts[0]["id"], boa_accounts[0]["id"], 100
)
