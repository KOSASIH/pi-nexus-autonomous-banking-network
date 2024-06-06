import zkproof

def generate_zkp_snarks_proof(transaction, private_key):
    proof = zkproof.generate_proof(transaction, private_key, zkproof.zk_snarks)
    return proof

def verify_zkp_snarks_proof(proof, public_key):
    zkproof.verify_proof(proof, public_key, zkproof.zk_snarks)
    return True
