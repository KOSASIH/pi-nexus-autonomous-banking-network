import numpy as np
from Pyfhel import Pyfhel, PyPtxt, PyCtxt

def he_encrypt(transaction, public_key):
    HE = Pyfhel()
    HE.contextGen(p=65537, m=8192, flagBatching=True)
    pub_key = HE.publicKeyGen()
    pub_key.Restore(public_key)
    ctxt = HE.encrypt(transaction, pub_key)
    return ctxt

def he_decrypt(ctxt, private_key):
    HE = Pyfhel()
    priv_key = HE.privateKeyGen()
    priv_key.Restore(private_key)
    ptxt = HE.decrypt(ctxt, priv_key)
    return ptxt.decrypt()

def he_compute(ctxt1, ctxt2):
    HE = Pyfhel()
    ctxt_result = HE.multiply(ctxt1, ctxt2)
    return ctxt_result
