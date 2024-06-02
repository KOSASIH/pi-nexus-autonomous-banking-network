import paypalrestsdk


class PayPalPaymentGateway:
    def __init__(self, client_id, client_secret, mode="sandbox"):
        self.client_id = client_id
        self.client_secret = client_secret
        self.mode = mode
        paypalrestsdk.configure(
            {"mode": mode, "client_id": client_id, "client_secret": client_secret}
        )

    def create_payment(self, amount, currency, intent="sale"):
        payment = paypalrestsdk.Payment(
            {
                "intent": intent,
                "payer": {"payment_method": "paypal"},
                "redirect_urls": {
                    "return_url": "http://return.url",
                    "cancel_url": "http://cancel.url",
                },
                "transactions": [
                    {
                        "item_list": {
                            "items": [
                                {
                                    "name": "Item Name",
                                    "sku": "Item SKU",
                                    "price": str(amount),
                                    "currency": currency,
                                    "quantity": 1,
                                }
                            ]
                        },
                        "amount": {"currency": currency, "total": str(amount)},
                        "description": "This is the payment description.",
                    }
                ],
            }
        )
        return payment

    def execute_payment(self, payment_id, payer_id):
        payment = paypalrestsdk.Payment.find(payment_id)
        payment.execute({"payer_id": payer_id})
        return payment
