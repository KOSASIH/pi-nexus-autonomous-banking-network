import hashlib


def calculate_hash(block):
    """Calculate the hash of a block"""
    block_string = str(block)
    return hashlib.sha256(block_string.encode()).hexdigest()


def is_valid_proof(block, proof):
    """Verify that a proof of work is valid"""
    guess = f'{block["index"]}{block["previous_hash"]}{proof}'.encode()
    guess_hash = hashlib.sha256(guess).hexdigest()
    return guess_hash[:4] == "0000"


def verify_chain(chain):
    """Verify that the chain is valid"""
    for i in range(1, len(chain)):
        current_block = chain[i]
        previous_block = chain[i - 1]

        if current_block["index"] != i:
            return False

        if current_block["previous_hash"] != calculate_hash(previous_block):
            return False

        if not is_valid_proof(current_block, current_block["proof"]):
            return False

    return True
