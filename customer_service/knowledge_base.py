class KnowledgeBase:
    def __init__(self):
        self.knowledge_base = {
            'account': {
                'open': 'To open a new account, please visit our website and fill out the account opening form.',
                'close': 'To close an existing account, please contact our customer support team.',
                'balance': 'To check your account balance, please log in to your account on our website or mobile app.'
            },
            'transaction': {
                'make': 'To make a new transaction, please log in to your account on our website or mobile app.',
                'history': 'To view your transaction history, please log in to your account on our website or mobile app.'
            },
            'other': {
                'help': 'For assistance, please contact our customer support team.'
            }
        }

    def get_response(self, intent):
        if any(intent.values()):
            for key, value in intent.items():
                if value:
                    return self.knowledge_base[key]
        else:
            return self.knowledge_base['other']
