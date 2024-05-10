import re

class TransactionValidation:
    def __init__(self):
        self.validation_rules = {
            'account_number': {
                'regex': r'^\d{10}$',
                'message': 'Invalid account number format. Please enter a 10-digit account number.'
            },
            'amount': {
                'min': 0,
                'max': 1000000,
                'message': 'Transaction amount must be between 0 and 1,000,000.'
            }
        }

    def validate_transaction(self, transaction):
        errors = []
        for field, rules in self.validation_rules.items():
            if field in transaction:
                for rule in rules:
                    if rule == 'regex':
                        if not re.match(rules[rule], transaction[field]):
                            errors.append(rules['message'])
                    elif rule == 'min':
                        if float(transaction[field]) < rules[rule]:
                            errors.append(f'Transaction amount must be at least {rules[rule]}')
                    elif rule == 'max':
                        if float(transaction[field]) > rules[rule]:
                            errors.append(f'Transaction amount must not exceed {rules[rule]}')
        return errors
