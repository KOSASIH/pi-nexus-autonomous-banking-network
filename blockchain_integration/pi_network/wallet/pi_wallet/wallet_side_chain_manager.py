import json
import time
from queue import Queue

class SideChainManager:
    def __init__(self, side_chain_url, main_chain_url):
        self.side_chain_url = side_chain_url
        self.main_chain_url = main_chain_url
        self.transaction_queue = Queue()

    def process_transactions(self):
        while True:
            # Get transactions from main chain
            main_chain_response = requests.get(self.main_chain_url + '/get_transactions')
            transactions = main_chain_response.json()

            # Add transactions to queue
            for transaction in transactions:
                self.transaction_queue.put(transaction)

            # Process transactions in queue
            while not self.transaction_queue.empty():
                transaction = self.transaction_queue.get()
                self.process_transaction(transaction)

            # Wait for 1 minute before processing next batch of transactions
            time.sleep(60)

    def process_transaction(self, transaction):
        # Send transaction to side chain
        side_chain_response = requests.post(self.side_chain_url, json=transaction)
        side_chain_tx_hash = side_chain_response.json()['tx_hash']

        # Wait for transaction to be confirmed on side chain
        while True:
            side_chain_status_response = requests.get(self.side_chain_url + '/' + side_chain_tx_hash)
            side_chain_status = side_chain_status_response.json()['status']
            if side_chain_status == 'confirmed':
                break

        print('Transaction processed on side chain:', side_chain_tx_hash)

if __name__ == '__main__':
    side_chain_url = 'http://localhost:3000/side_chain'
    main_chain_url = 'http://localhost:3000/main_chain'
    side_chain_manager = SideChainManager(side_chain_url, main_chain_url)

    side_chain_manager.process_transactions()
