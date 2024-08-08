# Polynomial utility functions for NTRU
import numpy as np

def poly_add(a, b):
    # Add two polynomials
    return (a + b) % params.ntru_q

def poly_mul(a, b):
    # Multiply two polynomials
    return (a * b) % params.ntru_q
