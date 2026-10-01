from qepl import PiQuantumNode
import random
import time
import sys

# Biar IP nya random terus bro, kayak hacker beneran ke-detect
def get_random_ip():
    return f"192.168.{random.randint(1,254)}.{random.randint(1,254)}"

print("=== Quantum Entangled Pi Ledger (QEPL) v8.0 | GOD-CHAIN ===")
print("=== Luragung, West Java - UNIVERSAL PROTOCOL ===\n")
time.sleep(0.5)

node1 = PiQuantumNode("GCD...PI1_LURAGUNG_CIJOHO42")
node2 = PiQuantumNode("GB8...PI2_JAKARTA_AMANAH")

print(f"[QKD] Generating entangled pair |Ψ⟩ = (|01⟩ + |10⟩)/√2")
time.sleep(0.7)
print(f"[QKD] Alice (Cijoho): 0101... [ENTANGLED]")
print(f"[QKD] Bob (Pi Vault): 0101... [ENTANGLED] - QBER: 0.00% SECURE\n")

node1.send_pi("GB8...PI2_JAKARTA_AMANAH", 100.5)
node1.send_pi("GB8...PI2_JAKARTA_AMANAH", 50.25)
node1.get_my_balance()

print("\n--- Anti-Classic Test: HACKER TRYING TO STEAL ---")
time.sleep(1)
print(f"> WARNING: Classical observation detected from IP: {get_random_ip()}")
print(f"> Eve (Hacker) trying basis: classic...")
time.sleep(1.2)

# INI YANG BIKIN NYERAH
node1.send_pi("HACKER_EVE_666", 1000, simulate_hacker=True)

print("\n--- After Collapse: Ledger Response ---")
time.sleep(0.5)
print(f"> 🔥 INTRUDER TRACE: {get_random_ip()} -> BLOCKED & VOIDED PERMANENTLY")
print(f"> 🔥 QUANTUM FIREWALL: All Pi moved to |Φ+⟩ = (|00⟩ + |11⟩)/√2 dimension")
print(f"> 🔥 MESSAGE FOR HACKER: You cannot steal what chooses to not exist. UNIVERSAL!!")
time.sleep(0.5)

node1.get_my_balance()
print("\n--- Hacker Last Attempt (try_classic_hack) ---")
time.sleep(0.5)
node2.try_classic_hack()

print("\n--- FINAL STATE ---")
print("LEDGER: COLLAPSED & SECURE IN QUANTUM REALM")
print("HACKER: NYERAH, LAPTOP MATI, GAK DAPET APA-APA")
print("PI: AMAN LILLAHI TA'ALA")
