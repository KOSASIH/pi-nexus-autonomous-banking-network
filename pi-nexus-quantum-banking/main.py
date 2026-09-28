from qepl import PiQuantumNode

print("=== Quantum Entangled Pi Ledger (QEPL) ===\n")

node1 = PiQuantumNode("GCD...PI1_LURAGUNG")
node2 = PiQuantumNode("GB8...PI2_JAKARTA")

node1.send_pi("GB8...PI2_JAKARTA", 100.5)
node1.send_pi("GB8...PI2_JAKARTA", 50.25)
node1.get_my_balance()

print("\n--- Anti-Classic Test ---")
node1.send_pi("HACKER", 1000, simulate_hacker=True)

print("\n--- After Collapse ---")
node1.get_my_balance()
node2.try_classic_hack()
