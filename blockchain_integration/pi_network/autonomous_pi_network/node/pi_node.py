import asyncio
import web3

class PINode:
    def __init__(self, node_address, contract_address):
        self.node_address = node_address
        self.contract_address = contract_address
        self.w3 = web3.Web3(web3.providers.AsyncHTTPProvider('https://mainnet.infura.io/v3/YOUR_PROJECT_ID'))

    async def register_node(self):
        #...
        tx_hash = self.w3.eth.send_transaction({'from': self.node_address, 'to': self.contract_address, 'data': '0x...'})
        await self.w3.eth.wait_for_transaction_receipt(tx_hash)

    async def get_node_list(self):
        #...
        node_list = self.w3.eth.call({'to': self.contract_address, 'data': '0x...'}).decode('utf-8')
        return node_list.split(',')
