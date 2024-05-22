from currency.exchange import ExchangeRate


class Transaction:
    def __init__(self, sender, receiver, amount, currency, fee=0.01):
        self.sender = sender
        self.receiver = receiver
        self.amount = amount
        self.currency = currency
        self.fee = fee
        self.exchange_rate = ExchangeRate(
            Currency("USD", "US Dollar", "$"), Currency("EUR", "Euro", "€"), 0.85
        )
