import asyncio
from stellar_sdk import Server, Listener
from eventlet import Event

class BlockchainEventListenerEDR:
    def __init__(self, horizon_url, network_passphrase):
        self.horizon_url = horizon_url
        self.network_passphrase = network_passphrase
        self.server = Server(horizon_url)
        self.listener = Listener(self.server)
        self.event_bus = Event()

    async def listen_for_events(self):
        async with self.listener:
            async for event in self.listener.stream():
                if event.type == 'transaction':
                    # Process transaction event
                    self.event_bus.send('transaction', event.transaction_hash)
                elif event.type == 'account_created':
                    # Process account created event
                    self.event_bus.send('account_created', event.account_id)

    def start_listening(self):
        asyncio.run(self.listen_for_events())

    def subscribe_to_event(self, event_type, callback):
        self.event_bus.on(event_type, callback)
