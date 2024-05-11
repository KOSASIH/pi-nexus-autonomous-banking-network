import datetime

class CreditCard:
    def __init__(self, card_number, limit):
        self.card_number = card_number
        self.limit = limit
        self.balance = 0
        self.interest_rate = 0.05
        self.closed = False

def process_payments():
    for card in all_cards:
        amount_due = card.balance * (1 + card.interest_rate)
        if amount_due > card.limit:
            print(f"WARNING: Credit limit exceeded for card {card.card_number}")
        else:
            card.balance = 0

def process_purchases():
    for card in all_cards:
        if card.closed:
            continue
        purchase_amount = float(input(f"Enter purchase amount for card {card.card_number}: "))
        if purchase_amount > card.limit - card.balance:
            print(f"PURCHASE DENIED: Insufficient credit for card {card.card_number}")
        else:
            card.balance += purchase_amount
            if card.balance > card.limit:
                print(f"WARNING: Credit limit exceeded for card {card.card_number}")

def set_credit_limit(card, limit):
    card.limit = limit

def get_credit_limit(card):
    return card.limit

def get_balance(card):
    return card.balance

def apply_interest(card):
    card.balance *= (1 + card.interest_rate)

def deny_credit(card):
    card.limit = 0
    card.closed = True

# Initialize list of all credit cards
all_cards = []
