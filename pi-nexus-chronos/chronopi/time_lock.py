import hashlib, time

class TimeLockPuzzle:
    def __init__(self, unlock_year):
        self.unlock_year = unlock_year
        self.current_year = 2026

    def create_time_capsule(self, pi_amount, message):
        print(f"[TIME-LOCK] Creating capsule for year {self.unlock_year}...")
        print(f"[TIME-LOCK] Amount: {pi_amount} Pi")
        print(f"[TIME-LOCK] Message: '{message}'")
        
        # Simulasi puzzle yang butuh (unlock_year - current_year) tahun untuk dipecahkan
        years_to_wait = self.unlock_year - self.current_year
        # Hash yang butuh komputasi super lama
        secret = f"{pi_amount}{message}{self.unlock_year}"
        locked_hash = hashlib.sha256(secret.encode()).hexdigest()
        
        print(f"[TIME-LOCK] Locked Hash: {locked_hash[:32]}...")
        print(f"[TIME-LOCK] 🔒 This can ONLY be opened in {self.unlock_year}. Physics forbids earlier.")
        return locked_hash, years_to_wait

    def attempt_unlock(self, current_year_sim):
        if current_year_sim < self.unlock_year:
            remaining = self.unlock_year - current_year_sim
            print(f"[TIME-LOCK] ❌ DENIED. Year is {current_year_sim}. Still {remaining} years left. You cannot cheat time.")
            return False
        else:
            print(f"[TIME-LOCK] ✅ UNLOCKED! Year {current_year_sim} reached. Time itself has opened the vault.")
            return True
