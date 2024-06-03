import json

class InteroperabilityProtocol:
    def __init__(self, main_chain_url, side_chain_url):
        self.main_chain_url = main_chain_url
        self.side_chain_url = side_chain_url

    def send_transaction(self, transaction):
        # Serialize transaction
        serialized_transaction = json.dumps(transaction)

        # Encrypt transaction
        encrypted_transaction = self.encrypt(serialized_transaction)

        # Send transaction to main chain
        main_chain_response = requests.post(self.main_chain_url, json=encrypted_transaction)
        main_chain_tx_hash = main_chain_response.json()['tx_hash']

        # Send transaction to side chain
        side_chain_response = requests.post(self.side_chain_url, json=encrypted_transaction)
        side_chain_tx_hash = side_chain_response.json()['tx_hash']

        return main_chain_tx_hash, side_chain_tx_hash

    def encrypt(self, data):
        # Implement encryption logic here
        # For demonstration purposes, assume encryption is successful
        return data

    def decrypt(self, data):
        # Implement decryption logic here
        # For demonstration purposes, assume decryption is successful
        return data

if __name__ == '__main__':
    main_chain_url = 'http://localhost:3000/main_chain'
    side_chain_url = 'http://localhost:3000/side_chain'
    interoperability_protocol = InteroperabilityProtocol(main_chain_url, side_chain_url)

    transaction = {
        'ender': 'address1',
        'ecipient': 'address2',
        'amount': 100,
        'fee': 10,
        'ignature': 'ignature'
    }

    main_chain_tx_hash, side_chain_tx_hash = interoperability_protocol.send_transaction(transaction)
    print('Main Chain Tx Hash:', main_chain_tx_hash)
    print('Side Chain Tx Hash:', side_chain_tx_hash)
