import braintree

class BraintreePaymentGateway:
    def __init__(self, merchant_id, public_key, private_key):
        self.merchant_id = merchant_id
        self.public_key = public_key
        self.private_key = private_key
        braintree.Configuration.configure(merchant_id, public_key, private_key)

    def create_payment(self, amount, currency):
        payment = braintree.Transaction.sale({
            'amount': amount,
            'currency': currency
        })
        return payment
