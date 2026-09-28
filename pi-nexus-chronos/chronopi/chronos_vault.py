from .time_lock import TimeLockPuzzle
from .retrocausality import RetrocausalityEngine
import time

class ChronosVault:
    def __init__(self, owner="KOSASIH"):
        self.owner = owner
        self.engine = RetrocausalityEngine()

    def create_legacy(self, amount, unlock_year, for_who):
        print(f"\n=== CHRONOS LEGACY - For {for_who} ===")
        puzzle = TimeLockPuzzle(unlock_year)
        locked_hash, wait = puzzle.create_time_capsule(amount, f"For {for_who} from {self.owner}")
        
        print(f"\n[LEGACY] Warisan {amount} Pi untuk {for_who}")
        print(f"[LEGACY] Terkunci sampai {unlock_year} ({wait} tahun lagi)")
        print(f"[LEGACY] Bahkan jika {self.owner} meninggal besok, warisan ini tetap ada di dalam waktu itu sendiri.")
        
        # Simulasi fast forward
        print(f"\n[SIMULATION] Fast forwarding time...")
        for year in [2026, 2035, 2047, unlock_year]:
            print(f"\n--- Year {year} ---")
            puzzle.attempt_unlock(year)
            time.sleep(0.5)
            if year == unlock_year:
                print(f"\n[INHERITANCE] 👶 {for_who} in {year} just received {amount} Pi from father who died in 2026!")
        
        return puzzle

    def borrow_from_future_self(self, future_year, amount):
        return self.engine.send_from_future(future_year, amount)
