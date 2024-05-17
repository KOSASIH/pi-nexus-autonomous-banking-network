import time
from datetime import datetime
from threading import Thread

class Exchange:
    def __init__(self):
        self.orders = []
        self.threads = []

    def add_order(self, order):
        self.orders.append(order)

    def start_threads(self):
        for order in self.orders:
            thread := Thread(target=self.execute_order, args=(order,))
            thread.start()
            self.threads.append(thread)

        for thread in self.threads:
            thread.join()

    def execute_order(self, order):
        # Implement the logic to execute the order
        # This will depend on the specific API or interface provided by the exchange
        # For now, we'll just print a message to the console
        print(f"Executing order: {order}")

class Order:
    def __init__(self, order_id, coin, quantity, price):
        self.order_id = order_id
        self.coin = coin
        self.quantity = quantity
        self.price = price

class Coin:
    def __init__(self, name, price):
        self.name = name
        self.price = price

if __name__ == "__main__":
    # Create an exchange object
    exchange := Exchange()

    # Create some orders
    order1 := Order(1, Coin("Pi Coin", 314.159), 100, 1.0)
    order2 := Order(2, Coin("Bitcoin", 50000.0), 5, 0.1)

    # Add the orders to the exchange
    exchange.add_order(order1)
    exchange.add_order(order2)

    # Start the order execution threads
    exchange.start_threads()
