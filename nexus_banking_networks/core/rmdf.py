import asyncio
import websockets

class RMDF:
    def __init__(self, market_data_url):
        self.market_data_url = market_data_url

    async def subscribe(self):
        async with websockets.connect(self.market_data_url) as websocket:
            await websocket.send("Subscribe")
            while True:
                message = await websocket.recv()
                print(message)

    async def fetch_data(self):
        async with websockets.connect(self.market_data_url) as websocket:
            await websocket.send("Fetch")
            data = await websocket.recv()
            return data
