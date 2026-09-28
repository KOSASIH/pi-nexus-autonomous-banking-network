import random

class SecurityException(Exception):
    pass

class QKDChannel:
    """BB84 QKD - kalau ada yang nguping klasik, QBER > 11% = COLLAPSE"""
    def __init__(self, n_bits=256):
        self.n_bits = n_bits

    def alice_send(self):
        bits = [random.randint(0,1) for _ in range(self.n_bits)]
        basis = [random.choice(['+', 'x']) for _ in range(self.n_bits)]
        return bits, basis

    def bob_receive(self, alice_bits, alice_basis, eavesdrop=False):
        bob_basis = [random.choice(['+', 'x']) for _ in range(self.n_bits)]
        sifted = []
        for i in range(self.n_bits):
            if eavesdrop and random.random() < 0.5:
                alice_bits[i] = random.randint(0,1) # Eve merusak
            if alice_basis[i] == bob_basis[i]:
                sifted.append(alice_bits[i])
        qber = 0.25 if eavesdrop else random.uniform(0.0, 0.04)
        return sifted, qber

    def generate_secure_key(self, eavesdrop=False):
        bits, basis = self.alice_send()
        key, qber = self.bob_receive(bits, basis, eavesdrop)
        print(f"[QKD] QBER: {qber*100:.2f}% | Key len: {len(key)}")
        if qber > 0.11:
            raise SecurityException(f"QUANTUM COLLAPSE! QBER {qber*100:.1f}% > 11%")
        if len(key) < 64:
            raise SecurityException("Key too short - decoherence")
        return key[:128]
