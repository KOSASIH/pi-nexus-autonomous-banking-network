import asyncio
from aiohttp import ClientSession
from json import dumps, loads

class RTP:
    def __init__(self, api_endpoint, api_key):
        self.api_endpoint = api_endpoint
        self.api_key = api_key
        self.session = ClientSession()

    async def process_transaction(self, transaction_data):
        headers = {'Authorization': f'Bearer {self.api_key}', 'Content-Type': 'application/json'}
        async with self.session.post(self.api_endpoint, headers=headers, data=dumps(transaction_data)) as response:
            response_data = await response.text()
            return loads(response_data)

    async def process_transactions(self, transactions_data):
        tasks = [self.process_transaction(data) for data in transactions_data]
        results = await asyncio.gather(*tasks)
        return results
