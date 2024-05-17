import json
import os
import time

import requests

# Configuration
BANKS_LIST = ["bank1.com", "bank2.com", "bank3.com"]
API_KEYS = ["key1", "key2", "key3"]


# Helper functions
def get_balance(bank_url, api_key):
    headers = {"Authorization": f"Bearer {api_key}"}
    response = requests.get(f"{bank_url}/api/balance", headers=headers)
    return response.json()["balance"]


def get_all_balances():
    all_balances = []
    for i in range(len(BANKS_LIST)):
        balance = get_balance(BANKS_LIST[i], API_KEYS[i])
        all_balances.append(balance)
    return all_balances


def main():
    while True:
        all_balances = get_all_balances()
        print(f"Current balances: {all_balances}")
        time.sleep(60)


if __name__ == "__main__":
    main()
