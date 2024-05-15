import asyncio

from aiohttp import ClientSession


class TransactionProcessor:
    def __init__(self):
        self.session = ClientSession()

    async def process_transaction(self, transaction_data):
        # Use asyncio to process transactions in parallel
        async with self.session.post(
            "https://api.example.com/transactions", json=transaction_data
        ) as response:
            return await response.json()

    async def process_transactions(self, transactions):
        tasks = [self.process_transaction(tx) for tx in transactions]
        results = await asyncio.gather(*tasks)
        return results
