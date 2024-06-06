import zkproof

def generate_zkp_proof(transaction, private_key):
    proof = zkproof.generate_proof(transaction, private_key)
    return proof

def verify_zkp_proof(proof, public_key):
    zkproof.verify_proof(proof, public_key)
    return True
