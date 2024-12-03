class Token:
    def __init__(self, name, symbol, total_supply):
        self.name = name
        self.symbol = symbol
        self.total_supply = total_supply
        self.balances = {}

    def mint(self, to_address, amount):
        if to_address not in self.balances:
            self.balances[to_address] = 0
        self.balances[to_address] += amount
        self.total_supply += amount
        print(f"Minted {amount} {self.symbol} to {to_address}. New balance: {self.balances[to_address]}")

    def transfer(self, from_address, to_address, amount):
        if from_address not in self.balances or self.balances[from_address] < amount:
            raise ValueError("Insufficient balance.")
        if to_address not in self.balances:
            self.balances[to_address] = 0
        self.balances[from_address] -= amount
        self.balances[to_address] += amount
        print(f"Transferred {amount} {self.symbol} from {from_address} to {to_address}.")

    def get_balance(self, address):
        return self.balances.get(address, 0)
