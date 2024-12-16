import requests

class CrossChain:
    def __init__(self, blockchain):
        self.blockchain = blockchain

    @staticmethod
    def send_transaction(recipient, amount):
        # Send a transaction to another blockchain
        transaction_data = {
            'sender': str(uuid4()),  # Simulate sender address
            'recipient': recipient,
            'amount': amount,
        }
        response = requests.post('http://other-blockchain-url/transactions/new', json=transaction_data)
        return response.json()

    @staticmethod
    def fetch_chain(node):
        response = requests.get(f'http://{node}/chain')
        if response.status_code == 200:
            return response.json()
        return None

    def synchronize_chains(self):
        for node in self.blockchain.nodes:
            chain = self.fetch_chain(node)
            if chain and self.blockchain.valid_chain(chain):
                self.blockchain.chain = chain
