import asyncio
from aiohttp import ClientSession
from json import dumps, loads

class HPDLT:
    def __init__(self, nodes):
        self.nodes = nodes
        self.session = ClientSession()

    async def broadcast_transaction(self, transaction_data):
        tasks = [self._send_transaction(node, transaction_data) for node in self.nodes]
        await asyncio.gather(*tasks)

    async def _send_transaction(self, node_url, transaction_data):
        headers = {'Content-Type': 'application/json'}
        async with self.session.post(node_url, headers=headers, data=dumps(transaction_data)) as response:
            response_data = await response.text()
            print(f"Response from {node_url}: {response_data}")
