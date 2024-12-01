class Token:
    def __init__(self, name, symbol, total_supply):
        self.name = name
        self.symbol = symbol
        self.total_supply = total_supply
        self.balances = {}
        self.allowances = {}

    def transfer(self, sender, recipient, amount):
        if self.balances.get(sender, 0) >= amount:
            self.balances[sender] -= amount
            self.balances[recipient] = self.balances.get(recipient, 0) + amount
            return True
        return False

    def approve(self, owner, spender, amount):
        if owner not in self.allowances:
            self.allowances[owner] = {}
        self.allowances[owner][spender] = amount
        return True

    def transfer_from(self, spender, owner, recipient, amount):
        if self.allowances.get(owner, {}).get(spender, 0 ) >= amount and self.balances.get(owner, 0) >= amount:
            self.balances[owner] -= amount
            self.balances[recipient] = self.balances.get(recipient, 0) + amount
            self.allowances[owner][spender] -= amount
            return True
        return False

    def balance_of(self, owner):
        return self.balances.get(owner, 0)

    def total_supply(self):
        return self.total_supply

    def mint(self, recipient, amount):
        self.total_supply += amount
        self.balances[recipient] = self.balances.get(recipient, 0) + amount

    def burn(self, owner, amount):
        if self.balances.get(owner, 0) >= amount:
            self.balances[owner] -= amount
            self.total_supply -= amount
            return True
        return False
