import amazon_pay


class AmazonPayPaymentGateway:
    def __init__(self, seller_id, client_id, client_secret):
        self.seller_id = seller_id
        self.client_id = client_id
        self.client_secret = client_secret
        amazon_pay.Configuration.configure(seller_id, client_id, client_secret)

    def create_payment(self, amount, currency):
        payment = amazon_pay.Payment.create({"amount": amount, "currency": currency})
        return payment
