import samsung_pay

class SamsungPayPaymentGateway:
    def __init__(self, service_id, service_type):
        self.service_id = service_id
        self.service_type = service_type
        samsung_pay.Configuration.configure(service_id, service_type)

    def create_payment(self, amount, currency):
        payment = samsung_pay.Payment.create({
            'amount': amount,
            'currency': currency
        })
        return payment
