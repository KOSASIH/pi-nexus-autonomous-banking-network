class CreditCard:
    def __init__(self, credit_card_number, credit_limit, application):
        self.credit_card_number = credit_card_number
        self.credit_limit = credit_limit
        self.balance = 0
        self.application = application

    def make_payment(self, amount):
        # Process a payment for the specified amount
        self.balance -= amount

    def charge_purchase(self, amount):
        # Charge a purchase for the specified amount
        if amount > self.credit_limit - self.balance:
            # Deny the purchase if the applicant has insufficient credit
            return False

        self.balance += amount
        return True


def process_payments():
    # Process payments for all issued credit cards
    for credit_card in issued_credit_cards:
        # Assume that the credit card company charges a 3% fee for processing payments
        payment_amount = credit_card.balance * 0.03

        # Process the payment
        credit_card.make_payment(payment_amount)


def process_purchases():
    # Process purchases for all issued credit cards
    for credit_card in issued_credit_cards:
        # Assume that the credit card company charges a 2% fee for processing purchases
        purchase_amount = credit_card.balance * 0.02

        # Process the purchase
        success = credit_card.charge_purchase(purchase_amount)

        if not success:
            # Deny the purchase if the applicant has insufficient credit
            print(
                f"The purchase of {purchase_amount} was denied for credit card {credit_card.credit_card_number}."
            )


issued_credit_cards = []
