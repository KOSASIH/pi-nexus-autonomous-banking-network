from collections import defaultdict
import time

class Token:
    def __init__(self, name, symbol, total_supply):
        self.name = name
        self.symbol = symbol
        self.total_supply = total_supply
        self.balances = defaultdict(int)

    def transfer(self, from_address, to_address, amount):
        if self.balances[from_address] < amount:
            raise ValueError("Insufficient balance")
        self.balances[from_address] -= amount
        self.balances[to_address] += amount

    def mint(self, to_address, amount):
        self.balances[to_address] += amount
        self.total_supply += amount

class Order:
    def __init__(self, order_id, user, token, amount, price, order_type):
        self.order_id = order_id
        self.user = user
        self.token = token
        self.amount = amount
        self.price = price
        self.order_type = order_type  # 'buy' or 'sell'
        self.timestamp = time.time()

class DEX:
    def __init__(self):
        self.order_book = defaultdict(list)  # token -> list of orders
        self.order_id_counter = 0

    def place_order(self, user, token, amount, price, order_type):
        if order_type not in ['buy', 'sell']:
            raise ValueError("Order type must be 'buy' or 'sell'")
        order = Order(self.order_id_counter, user, token, amount, price, order_type)
        self.order_book[token].append(order)
        self.order_id_counter += 1
        print(f"Placed {order_type} order: {amount} {token.symbol} at {price} by {user}")

        # Attempt to match orders
        self.match_orders(token)

    def match_orders(self, token):
        buy_orders = sorted([o for o in self.order_book[token] if o.order_type == 'buy'], key=lambda x: -x.price)
        sell_orders = sorted([o for o in self.order_book[token] if o.order_type == 'sell'], key=lambda x: x.price)

        while buy_orders and sell_orders:
            buy_order = buy_orders[0]
            sell_order = sell_orders[0]

            if buy_order.price >= sell_order.price:
                trade_amount = min(buy_order.amount, sell_order.amount)
                trade_price = sell_order.price

                # Execute trade
                print(f"Executed trade: {trade_amount} {token.symbol} at {trade_price} between {buy_order.user} and {sell_order.user}")

                # Update order amounts
                buy_order.amount -= trade_amount
                sell_order.amount -= trade_amount

                # Remove completed orders
                if buy_order.amount == 0:
                    buy_orders.pop(0)
                if sell_order.amount == 0:
                    sell_orders.pop(0)

            else:
                break  # No more matches possible

    def cancel_order(self, user, order_id, token):
        for order in self.order_book[token]:
            if order.order_id == order_id and order.user == user:
                self.order_book[token].remove(order)
                print(f"Cancelled order {order_id} for {user}")
                return
        print("Order not found or not authorized to cancel.")

# Example usage
if __name__ == "__main__":
    # Create tokens
    token_a = Token("TokenA", "TKA", 1000000)
    token_b = Token("TokenB", "TKB", 1000000)

    # Mint some tokens for the user
    token_a.mint("user1", 10000)
    token_b.mint("user2", 10000)

    # Create a DEX
    dex = DEX()

    # User places orders
    dex.place_order("user1", token_a, 100, 10, 'buy')  # User1 wants to buy 100 TKA at 10
    dex.place_order("user2", token_a, 100, 9, 'sell')  # User 2 wants to sell 100 TKA at 9

    # User places another order
    dex.place_order("user1", token_a, 50, 11, 'buy')  # User1 wants to buy 50 TKA at 11
    dex.place_order("user2", token_a, 50, 10, 'sell')  # User2 wants to sell 50 TKA at 10

    # User cancels an order
    dex.cancel_order("user1", 0, token_a)  # User1 cancels their first buy order

    # User places a new order
    dex.place_order("user2", token_a, 30, 8, 'sell')  # User2 wants to sell 30 TKA at 8

    # Check remaining orders
    print("Remaining orders for TokenA:")
    for order in dex.order_book[token_a]:
        print(f"Order ID: {order.order_id}, User: {order.user}, Amount: {order.amount}, Price: {order.price}, Type: {order.order_type}")
