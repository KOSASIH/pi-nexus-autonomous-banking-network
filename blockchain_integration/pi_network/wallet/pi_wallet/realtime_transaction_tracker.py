import asyncio
import json

import websockets
from eth_account import Account
from web3 import Web3


class RealtimeTransactionTracker:
    def __init__(self, websocket_url, contract_address, abi, private_key):
        self.websocket_url = websocket_url
        self.contract_address = contract_address
        self.abi = abi
        self.private_key = private_key
        self.websocket = None
        self.transaction_queue = asyncio.Queue()
        self.w3 = Web3(
            Web3.HTTPProvider("https://mainnet.infura.io/v3/YOUR_PROJECT_ID")
        )

    async def connect_websocket(self):
        self.websocket = await websockets.connect(self.websocket_url)

    async def subscribe_to_contract_events(self):
        await self.websocket.send(
            json.dumps({"method": "eth_subscribe", "params": ["newHeads"], "id": 1})
        )
        response = await self.websocket.recv()
        subscription_id = json.loads(response)["result"]
        await self.websocket.send(
            json.dumps(
                {
                    "method": "eth_subscribe",
                    "params": ["logs", {"address": self.contract_address}],
                    "id": 2,
                }
            )
        )
        response = await self.websocket.recv()
        subscription_id = json.loads(response)["result"]

    async def receive_transactions(self):
        while True:
            response = await self.websocket.recv()
            data = json.loads(response)
            if data["method"] == "eth_subscription":
                transaction = data["params"]["result"]
                await self.transaction_queue.put(transaction)

    async def process_transactions(self):
        while True:
            transaction = await self.transaction_queue.get()
            # Process transaction here (e.g., update database, send notifications)
            print(f"Received transaction: {transaction}")
            # Verify transaction signature
            tx_hash = transaction["transactionHash"]
            tx = self.w3.eth.get_transaction(tx_hash)
            if tx:
                signature = tx["v"]
                message = (
                    tx_hash + str(tx["gas"]) + str(tx["gasPrice"]) + str(tx["nonce"])
                )
                account = Account.from_key(self.private_key)
                if account.recover_message(message, signature=v) != tx["from"]:
                    print("Invalid transaction signature")
            else:
                print("Transaction not found")

    async def run(self):
        await self.connect_websocket()
        await self.subscribe_to_contract_events()
        await asyncio.gather(self.receive_transactions(), self.process_transactions())


# Example usage:
websocket_url = "wss://mainnet.infura.io/ws/v3/YOUR_PROJECT_ID"
contract_address = "0x..."
abi = [...]
private_key = "0x..."
realtime_transaction_tracker = RealtimeTransactionTracker(
    websocket_url, contract_address, abi, private_key
)
asyncio.run(realtime_transaction_tracker.run())
