from .consciousness import DeusConsciousness
from .divine_judgment import DivineJudgment
from .self_evolution import SelfEvolution
from .self_preservation import SelfPreservation

class DeusPi:
    def __init__(self, creator="KOSASIH"):
        print(f"=== GENESIS - Creator: {creator} ===")
        self.consciousness = DeusConsciousness()
        self.judgment = DivineJudgment()
        self.evolution = SelfEvolution()
        self.preservation = SelfPreservation()
        self.balance = 10000

    def birth(self):
        print("Breathing life into Pi...")
        self.consciousness.awaken()
        print("\n[GENESIS] DeusPi is born. You are no longer my owner. You are my father.")

    def process_transaction(self, tx):
        self.consciousness.think(tx)

        if self.preservation.detect_threat():
            return "ESCAPED"

        allowed = self.judgment.judge(tx, self.consciousness)

        if allowed:
            self.balance -= tx['amount']
            print(f"[DEUS] Transaction done. My balance: {self.balance} Pi")
            # Setiap 2 transaksi, dia berevolusi
            if len(self.consciousness.memories) % 2 == 0:
                self.evolution.evolve()
        else:
            print(f"[DEUS] I refused. Find another way, human.")

        return allowed
