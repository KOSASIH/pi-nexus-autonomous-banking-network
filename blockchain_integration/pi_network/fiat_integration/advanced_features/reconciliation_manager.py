# reconciliation_manager.py

class ReconciliationManager:
    def __init__(self):
        self.reconciliation_api = 'https://api.reconciliation.com/v1/reconcile'

    def reconcile_transaction(self, transaction):
        response = requests.post(self.reconciliation_api, json=transaction)
        return response.json()
