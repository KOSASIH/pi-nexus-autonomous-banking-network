import asyncio
import websockets

class MarketDataFeed:
    def __init__(self, market_data_url):
        self.market_data_url = market_data_url

    async def get_realtime_data(self):
        async with websockets.connect(self.market_data_url) as websocket:
            while True:
                data = await websocket.recv()
                # Process real-time market data
                print(data)

# Example usage:
market_data_url = 'wss://market-data-feed.com'
market_data_feed = MarketDataFeed(market_data_url)
asyncio.get_event_loop().run_until_complete(market_data_feed.get_realtime_data())
