import squareconnect

class SquarePaymentGateway:
    def __init__(self, access_token):
        self.access_token = access_token
        squareconnect.api_client.configuration.access_token = access_token

    def create_payment(self, amount, currency):
        transaction_api = squareconnect.TransactionsApi()
        request_body = {
            'amount_money': {
                'amount': int(amount * 100),
                'currency_code': currency
            },
            'payment_method_types': ['CARD'],
            'idempotency_key': squareconnect.util.random_string(32)
        }
        response = transaction_api.create_transaction(request_body)
        return response
