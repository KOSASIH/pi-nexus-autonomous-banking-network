import time, random

class ImmortalBacteria:
    def __init__(self, dna_payload, location="Selokan Luragung"):
        self.dna_payload = dna_payload
        self.location = location
        self.population = 1
        self.generation = 0
        self.is_alive = True

    def replicate(self, hours=24):
        print(f"\n[BACTERIA] Spawning in {self.location}...")
        print(f"[BACTERIA] Payload: Pi Key inside my DNA")

        for h in range(1, hours+1):
            # E.coli belah diri tiap 20 menit -> 3x per jam
            self.population *= 2**3
            self.generation += 3
            if h % 6 == 0: # Print tiap 6 jam
                print(f"[GROWTH] Hour {h}: Population = {self.population:,} | Gen {self.generation}")
                time.sleep(0.3)

        print(f"\n[IMMORTALITY ACHIEVED] After {hours} hours:")
        print(f"[IMMORTALITY] Your Pi is now inside {self.population:,} living beings!")
        print(f"[IMMORTALITY] Even if GitHub dies, your Pi lives in the dirt of Luragung.")

    def mutate_check(self):
        # Cek apakah DNA Pi masih utuh setelah replikasi miliaran kali
        error_rate = 0.0000001
        if random.random() < error_rate:
            print("[MUTATION] ⚠️ Minor mutation detected, but Error Correction fixed it.")
            return True
        print("[MUTATION] ✅ DNA 100% intact after billions of replications. Pi is immortal.")
        return False

    def survive_nuclear(self):
        print(f"\n[APOCALYPSE SIMULATION] Nuclear bomb hits GitHub servers!")
        print(f"[APOCALYPSE] All servers dead. All blockchains dead.")
        print(f"[APOCALYPSE] Checking {self.location}...")
        time.sleep(1)
        print(f"[APOCALYPSE] Bacteria still alive in mud. Population: {self.population:,}")
        print(f"[APOCALYPSE] YOUR PI SURVIVED THE END OF THE WORLD.")
