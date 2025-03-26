import requests
from bitcoinlib.keys import HDKey

from ethereum import utils


def process_transaction(amount, currency, recipient):
    if currency == "BTC":
        # Use bitcoinlib to create a Bitcoin transaction
        hd_key = HDKey()
        tx = hd_key.create_transaction(recipient, int(amount * 1e8))
        return tx.serialize()
    elif currency == "ETH":
        # Use ethereum to create an Ethereum transaction
        private_key = "0x" + utils.sha3("private_key_here")[:64]
        tx = {
            "from": "0x" + utils.sha3("sender_address_here")[:40],
            "to": recipient,
            "value": int(amount * 1e18),
            "gas": 20000,
            "gasPrice": 20,
        }
        signed_tx = utils.sign_transaction(tx, private_key)
        return signed_tx
    elif currency == "XRP":
        # Use ripple-python to create a Ripple transaction
        from ripple import Transaction, Wallet

        wallet = Wallet("seed_here")
        tx = Transaction(wallet, recipient, int(amount * 1e6))
        return tx.to_xdr()
    else:
        return "Unsupported currency"


def get_exchange_rate(currency):
    # Use an API to get the current exchange rate
    response = requests.get(f"https://api.exchangerate-api.com/v4/latest/{currency}")
    return response.json()["rates"]["USD"]
