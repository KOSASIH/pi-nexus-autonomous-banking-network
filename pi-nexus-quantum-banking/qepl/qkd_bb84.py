class QuantumLedger:
    def __init__(self, entangled_pair_id):
        self.entangled_id = entangled_pair_id
        self.is_collapsed = False
        self.transactions = []

    def add_pi_transaction(self, sender, receiver, amount_pi, qkd_key):
        if self.is_collapsed:
            raise Exception("Ledger sudah collapse, tidak bisa dibaca node klasik!")
        # Encrypt amount dengan QKD key
        encrypted = amount_pi ^ sum(qkd_key[:8]) # simplified
        self.transactions.append((sender, receiver, encrypted))
        print(f"✓ {amount_pi} Pi terkunci dengan QKD")

    def measure_ledger(self, basis_attempt='classic'):
        if basis_attempt == 'classic':
            self.is_collapsed = True
            self.transactions = []
            return "LEDGER COLLAPSED - Hacker klasik terdeteksi, data hilang"
        return self.transactions
