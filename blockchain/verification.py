import hashlib

def verify_transaction(sender, receiver, amount, timestamp, previous_hash):
    transaction = {
        'sender': sender,
        'receiver': receiver,
        'amount': amount,
        'timestamp': timestamp,
        'previous_hash': previous_hash
    }
    transaction_string = str(transaction)
    transaction_hash = hashlib.sha256(transaction_string.encode()).hexdigest()

    if transaction_hash == previous_hash:
        return True
    else:
        return False
