import random

def issue_credit_card(application):
    # Define the minimum credit limit for credit card issuance
    MIN_CREDIT_LIMIT = 500

    # Define the maximum credit limit for credit card issuance
    MAX_CREDIT_LIMIT = 10000

    # Calculate the credit limit based on the applicant's income
    credit_limit = int(application.income * 0.02)

    # Set the minimum credit limit if the calculated credit limit is too low
    if credit_limit < MIN_CREDIT_LIMIT:
        credit_limit = MIN_CREDIT_LIMIT

    # Set the maximum credit limit if the calculated credit limit is too high
    if credit_limit > MAX_CREDIT_LIMIT:
        credit_limit = MAX_CREDIT_LIMIT

    # Generate a random credit card number
    credit_card_number = generate_credit_card_number()

    # Create a new credit card object
    credit_card = CreditCard(credit_card_number, credit_limit, application)

    # Add the credit card to the list of issued credit cards
    issued_credit_cards.append(credit_card)

    return credit_card

def generate_credit_card_number():
    # Generate a random 16-digit credit card number
    return random.randint(1000000000000000, 9999999999999999)
