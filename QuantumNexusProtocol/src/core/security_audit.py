import hashlib
import json
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class SecurityAudit:
    def __init__(self):
        self.transactions = []  # List to hold transactions
        self.vulnerabilities = []  # List to hold detected vulnerabilities

    def add_transaction(self, transaction):
        """Add a transaction to the audit log."""
        if self.validate_transaction(transaction):
            self.transactions.append(transaction)
            logging.info(f"Transaction added: {transaction}")
        else:
            logging.warning(f"Transaction failed validation: {transaction}")

    def validate_transaction(self, transaction):
        """Validate a transaction for common security issues."""
        # Check for missing fields
        required_fields = ['id', 'sender', 'recipient', 'amount', 'signature']
        for field in required_fields:
            if field not in transaction:
                self.vulnerabilities.append(f"Missing field: {field} in transaction {transaction['id']}")
                return False

        # Check for negative amounts
        if transaction['amount'] < 0:
            self.vulnerabilities.append(f"Negative amount in transaction {transaction['id']}")
            return False

        # Check for valid signature (placeholder for actual signature verification)
        if not self.verify_signature(transaction['signature']):
            self.vulnerabilities.append(f"Invalid signature in transaction {transaction['id']}")
            return False

        return True

    def verify_signature(self, signature):
        """Placeholder for signature verification logic."""
        # In a real implementation, you would verify the signature against the sender's public key
        return True  # Assume signature is valid for this example

    def log_vulnerabilities(self):
        """Log detected vulnerabilities."""
        if self.vulnerabilities:
            logging.info("Detected vulnerabilities:")
            for vulnerability in self.vulnerabilities:
                logging.warning(vulnerability)
        else:
            logging.info("No vulnerabilities detected.")

    def generate_transaction_hash(self, transaction):
        """Generate a hash for a transaction."""
        transaction_string = json.dumps(transaction, sort_keys=True).encode()
        return hashlib.sha256(transaction_string).hexdigest()

    def get_audit_report(self):
        """Generate a report of the audit."""
        report = {
            'total_transactions': len(self.transactions),
            'vulnerabilities_detected': len(self.vulnerabilities),
            'vulnerabilities': self.vulnerabilities
        }
        return report

# Example usage
if __name__ == '__main__':
    audit = SecurityAudit()

    # Simulate adding transactions
    transactions = [
        {'id': '1', 'sender': 'Alice', 'recipient': 'Bob', 'amount': 50, 'signature': 'sig1'},
        {'id': '2', 'sender': 'Bob', 'recipient': 'Charlie', 'amount': -20, 'signature': 'sig2'},  # Invalid
        {'id': '3', 'sender': 'Charlie', 'recipient': 'Alice', 'amount': 30, 'signature': 'sig3'},
        {'id': '4', 'sender': 'Alice', 'recipient': 'Bob', 'amount': 100}  # Missing signature
    ]

    for transaction in transactions:
        audit.add_transaction(transaction)

    # Log vulnerabilities
    audit.log_vulnerabilities()

    # Generate audit report
    report = audit.get_audit_report()
    print("Audit Report:", report)
