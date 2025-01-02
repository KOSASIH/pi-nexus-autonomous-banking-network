class FraudDetection:
    def __init__(self):
        self.transaction_history = []

    def log_transaction(self, payment):
        self.transaction_history.append(payment)

    @staticmethod
    def detect_fraud(payment):
        # Simple fraud detection logic based on transaction patterns
        if payment.amount > 10000:  # Example threshold
            print(f"Fraud alert! High transaction amount detected: {payment.amount} from {payment.payer}.")
            return True
        return False
