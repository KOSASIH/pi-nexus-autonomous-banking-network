import random, uuid
from dataclasses import dataclass

@dataclass
class EntangledPhoton:
    id: str
    pair_id: str
    polarization: int
    is_measured: bool = False

class EntanglementSource:
    """Buat Bell Pair |Φ+> = (|00> + |11>)/√2 - ini Proof-of-Entanglement"""
    def __init__(self):
        self.pairs = {}

    def create_bell_pair(self):
        pair_id = str(uuid.uuid4())[:8]
        state = random.randint(0,1)
        a = EntangledPhoton(f"photon-A-{pair_id}", pair_id, state)
        b = EntangledPhoton(f"photon-B-{pair_id}", pair_id, state)
        self.pairs[pair_id] = (a, b)
        print(f"[ENTANGLEMENT] Bell pair {pair_id} -> |{state}{state}>")
        return a, b

    def verify_entanglement(self, pair_id):
        return pair_id in self.pairs

quantum_source = EntanglementSource()
