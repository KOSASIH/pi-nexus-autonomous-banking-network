class OrderBook:
    def __init__(self):
        self.bids = []
        self.asks = []

    def add_bid(self, price, quantity):
        self.bids.append((price, quantity))

    def add_ask(self, price, quantity):
        self.asks.append((price, quantity))

    def get_best_bid(self):
        return max(self.bids, key=lambda x: x[0])

    def get_best_ask(self):
        return min(self.asks, key=lambda x: x[0])

    def match_orders(self):
        # implement order matching logic here
        pass
