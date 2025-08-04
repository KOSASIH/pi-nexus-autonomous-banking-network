import json
from hashlib import sha256

import requests


class CrossChainBridge:
    def __init__(self, main_chain_url, side_chain_url):
        self.main_chain_url = main_chain_url
        self.side_chain_url = side_chain_url

    def send_transaction(self, transaction):
        # Hash transaction
        tx_hash = sha256(json.dumps(transaction).encode()).hexdigest()

        # Send transaction to main chain
        main_chain_response = requests.post(self.main_chain_url, json=transaction)
        main_chain_tx_hash = main_chain_response.json()["tx_hash"]

        # Wait for transaction to be confirmed on main chain
        while True:
            main_chain_status_response = requests.get(
                self.main_chain_url + "/" + main_chain_tx_hash
            )
            main_chain_status = main_chain_status_response.json()["status"]
            if main_chain_status == "confirmed":
                break

        # Send transaction to side chain
        side_chain_response = requests.post(self.side_chain_url, json=transaction)
        side_chain_tx_hash = side_chain_response.json()["tx_hash"]

        # Wait for transaction to be confirmed on side chain
        while True:
            side_chain_status_response = requests.get(
                self.side_chain_url + "/" + side_chain_tx_hash
            )
            side_chain_status = side_chain_status_response.json()["status"]
            if side_chain_status == "confirmed":
                break

        return main_chain_tx_hash, side_chain_tx_hash

    def verify_transaction(self, tx_hash, chain_url):
        # Get transaction from chain
        response = requests.get(chain_url + "/" + tx_hash)
        transaction = response.json()

        # Verify transaction signature
        if self.verify_signature(
            transaction["sender"], transaction["signature"], transaction
        ):
            return True
        else:
            return False

    def verify_signature(self, sender, signature, transaction):
        # Implement signature verification logic here
        # For demonstration purposes, assume signature is valid
        return True


if __name__ == "__main__":
    main_chain_url = "http://localhost:3000/main_chain"
    side_chain_url = "http://localhost:3000/side_chain"
    cross_chain_bridge = CrossChainBridge(main_chain_url, side_chain_url)

    transaction = {
        "ender": "address1",
        "ecipient": "address2",
        "amount": 100,
        "fee": 10,
        "ignature": "ignature",
    }

    main_chain_tx_hash, side_chain_tx_hash = cross_chain_bridge.send_transaction(
        transaction
    )
    print("Main Chain Tx Hash:", main_chain_tx_hash)
    print("Side Chain Tx Hash:", side_chain_tx_hash)

    # Verify transaction on main chain
    if cross_chain_bridge.verify_transaction(main_chain_tx_hash, main_chain_url):
        print("Transaction verified on main chain")
    else:
        print("Transaction verification failed on main chain")

    # Verify transaction on side chain
    if cross_chain_bridge.verify_transaction(side_chain_tx_hash, side_chain_url):
        print("Transaction verified on side chain")
    else:
        print("Transaction verification failed on side chain")
