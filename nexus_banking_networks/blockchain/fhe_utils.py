import numpy as np
from FHE import FHE

def generate_fhe_keypair():
    fhe = FHE()
    private_key = fhe.keygen()
    public_key = fhe.get_public_key(private_key)
    return private_key, public_key

def fhe_encrypt(transaction, public_key):
    fhe = FHE()
    ctxt = fhe.encrypt(transaction, public_key)
    return ctxt

def fhe_decrypt(ctxt, private_key):
    fhe = FHE()
    ptxt = fhe.decrypt(ctxt, private_key)
    return ptxt

def fhe_compute(ctxt1, ctxt2):
    fhe = FHE()
    ctxt_result = fhe.multiply(ctxt1, ctxt2)
    return ctxt_result
