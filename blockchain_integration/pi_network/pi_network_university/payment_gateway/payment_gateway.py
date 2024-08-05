# payment_gateway.py
class PaymentGateway:
    def __init__(self):
        self.payment_methods = []

    def add_payment_method(self, payment_method):
        self.payment_methods.append(payment_method)

    def process_payment(self, payment_method, amount):
        # implement payment processing logic
        pass
