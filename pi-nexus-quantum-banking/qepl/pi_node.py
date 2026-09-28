from.entanglement import quantum_source
from.quantum_ledger import QuantumLedger

class PiQuantumNode:
    def __init__(self, pi_address):
        self.pi_address = pi_address
        photon_a, _ = quantum_source.create_bell_pair()
        self.photon = photon_a
        self.ledger = QuantumLedger(photon_a.id)
        print(f"[Pi Node] {pi_address} photon {photon_a.id}")

    def send_pi(self, to, amount, simulate_hacker=False):
        print(f"\n[Pi Tx] {self.pi_address} -> {to} : {amount} Pi")
        if simulate_hacker:
            print("[!] Hacker klasik intercept...")
        return self.ledger.add_pi_transaction(self.pi_address, to, amount, simulate_hacker)

    def get_my_balance(self):
        bal = self.ledger.get_balance(self.pi_address)
        print(f"[Balance] {bal} Pi (quantum-verified)")
        return bal

    def try_classic_hack(self):
        print("\n[HACK] Classic read...")
        print(self.ledger.read_ledger(basis='classic'))
