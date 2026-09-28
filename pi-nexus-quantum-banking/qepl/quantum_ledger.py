from.qkd_bb84 import QKDChannel, SecurityException
from.entanglement import quantum_source
import time, hashlib

class QuantumLedger:
    def __init__(self, photon_id: str):
        self.pair_id = photon_id.split('-')[-1]
        self.blocks = []
        self.is_collapsed = False
        self.qkd = QKDChannel()

    def _qhash(self, data, key):
        ks = ''.join(map(str, key[:32]))
        return hashlib.sha256((data + ks).encode()).hexdigest()

    def add_pi_transaction(self, sender, receiver, amount, eavesdrop_sim=False):
        if self.is_collapsed:
            print("❌ LEDGER COLLAPSED")
            return False
        if not quantum_source.verify_entanglement(self.pair_id):
            self.collapse("Entanglement lost")
            return False
        try:
            qkey = self.qkd.generate_secure_key(eavesdrop=eavesdrop_sim)
        except SecurityException as e:
            self.collapse(str(e))
            return False
        block = {
            'sender': sender, 'receiver': receiver,
            'amount_pi': amount,
            'qkd_hash': self._qhash(f"{sender}->{receiver}:{amount}:{time.time()}", qkey),
            'pair_id': self.pair_id,
            'quantum_secure': True
        }
        self.blocks.append(block)
        print(f"✓ Block #{len(self.blocks)} | {amount} Pi | {block['qkd_hash'][:12]}... SECURE")
        return True

    def collapse(self, reason):
        self.is_collapsed = True
        self.blocks = []
        print(f"💥 COLLAPSE: {reason} - Ledger voided!")

    def read_ledger(self, basis='quantum'):
        if basis == 'classic':
            self.collapse("Classic read attempt")
            return "VOID"
        return self.blocks if not self.is_collapsed else "VOID - collapsed"

    def get_balance(self, addr, basis='quantum'):
        if basis == 'classic' or self.is_collapsed:
            return 0
        bal = 0
        for b in self.blocks:
            if b['receiver'] == addr: bal += b['amount_pi']
            if b['sender'] == addr: bal -= b['amount_pi']
        return bal
