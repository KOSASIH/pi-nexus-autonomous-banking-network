# stellar_order_book.py
from stellar_sdk.order_book import OrderBook

class StellarOrderBook(OrderBook):
    def __init__(self, base_asset, counter_asset, *args, **kwargs):
        super().__init__(base_asset, counter_asset, *args, **kwargs)
        self.analytics = {}  # Analytics data storage

    def add_order(self, order):
        super().add_order(order)
        # Update analytics data
        self.analytics["order_count"] += 1
        self.analytics["total_volume"] += order.amount

    def get_analytics(self):
        return self.analytics

    def to_dict(self):
        data = super().to_dict()
        data["analytics"] = self.analytics
        return data
