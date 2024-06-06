import numpy as np
from pqc import Lattice, Ring, Poly

def generate_qr_lattice_keypair():
    lattice = Lattice(1024, 2)
    private_key = lattice.sample()
    public_key = lattice.get_public_key(private_key)
    return private_key, public_key

def qrc_lattice_encrypt(transaction, public_key):
    lattice = Lattice(1024, 2)
    ctxt = lattice.encrypt(transaction, public_key)
    return ctxt

def qrc_lattice_decrypt(ctxt, private_key):
    lattice = Lattice(1024, 2)
    ptxt = lattice.decrypt(ctxt, private_key)
    return ptxt

def qrc_lattice_sign(transaction, private_key):
    lattice = Lattice(1024, 2)
    signature = lattice.sign(transaction, private_key)
    return signature

def qrc_lattice_verify(transaction, signature, public_key):
    lattice = Lattice(1024, 2)
    lattice.verify(transaction, signature, public_key)
    return True
