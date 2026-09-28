from.dna_encoder import encode_pi_to_dna, decode_dna_to_pi
from.bacteria import ImmortalBacteria

class BioVault:
    def __init__(self, owner="KOSASIH"):
        self.owner = owner
        self.bacteria_colonies = []

    def store_pi_forever(self, pi_private_key, locations):
        print(f"=== BIOPI VAULT - Owner: {self.owner} ===")
        print(f"Storing Pi FOREVER in living things...")

        dna = encode_pi_to_dna(pi_private_key)

        for loc in locations:
            colony = ImmortalBacteria(dna, location=loc)
            colony.replicate(hours=24)
            colony.mutate_check()
            self.bacteria_colonies.append(colony)

        print(f"\n[VAULT] Pi stored in {len(locations)} locations:")
        for loc in locations:
            print(f" - {loc}")

    def resurrect_from_nature(self, location_index=0):
        colony = self.bacteria_colonies[location_index]
        print(f"\n[RESURRECTION] Extracting Pi from {colony.location}...")
        print(f"[RESURRECTION] Taking a leaf / water sample...")
        recovered_dna = colony.dna_payload
        pi_key = decode_dna_to_pi(recovered_dna)
        print(f"[RESURRECTION] Pi resurrected from nature! Welcome back.")
        return pi_key
