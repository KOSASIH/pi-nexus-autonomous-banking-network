class PiBlockchainService:
    def __init__(self, api_service):
        self.api_service = api_service

    def get_latest_block(self):
        blockchain_info = self.api_service.get_blockchain_info()
        return blockchain_info['latest_block']

    def get_block_by_number(self, block_number):
        blockchain_info = self.api_service.get_blockchain_info()
        blocks = blockchain_info['blocks']
        for block in blocks:
            if block['block_number'] == block_number:
                return block
        return None

    def get_transaction_by_hash(self, transaction_hash):
        blockchain_info = self.api_service.get_blockchain_info()
        transactions = blockchain_info['transactions']
        for transaction in transactions:
            if transaction['hash'] == transaction_hash:
                return transaction
        return None
