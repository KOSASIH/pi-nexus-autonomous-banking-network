import requests

def get_pi_network_balance(address):
    response = requests.get(f"https://api.pi.network/balance/{address}")
    return response.json()["balance"]
