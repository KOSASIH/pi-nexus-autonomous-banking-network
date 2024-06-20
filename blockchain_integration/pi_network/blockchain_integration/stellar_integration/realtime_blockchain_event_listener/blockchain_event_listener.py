import asyncio
from stellar_sdk import Server, Listener

class BlockchainEventListener:
    def __init__(self, horizon_url, network_passphrase):
        self.horizon_url = horizon_url
        self.network_passphrase = network_passphrase
        self.server = Server(horizon_url)
        self.listener = Listener(self.server)

    async def listen_for_events(self):
        async with self.listener:
            async for event in self.listener.stream():
                if event.type == 'transaction':
                    # Process transaction event
                    print(f"Transaction: {event.transaction_hash}")
                elif event.type == 'account_created':
                    # Process account created event
                    print(f"Account created: {event.account_id}")

    def start_listening(self):
        asyncio.run(self.listen_for_events())
