from neuropi import ConsciousnessWallet

print("=== NeuroPi - Your Mind is Your Bank ===")

wallet = ConsciousnessWallet("GCD...KOSASIH_BRAIN", heir_address="GB8...ANAK_LU")

# 1. Enroll otak
wallet.create_wallet()

# 2. Transaksi normal saat tenang
wallet.send_pi("GB8...WARUNG", 50)

# 3. Coba transaksi saat stres
print("\n--- Simulasi Lu Stres Kerja ---")
wallet.firewall.analyze_state = lambda: ("STRESSED", 26, 95)
wallet.send_pi("GB8...SHOPEE", 100)

# 4. PALING GILA: Ditodong orang, jantung naik
print("\n--- Simulasi Ditodong Pistol ---")
wallet.send_pi("HACKER_TODONG", 1000, forced_panic=True)

# 5. Meninggal -> waris
print("\n--- Simulasi Flatline ---")
wallet.firewall.analyze_state = lambda: ("DEAD_FLATLINE", 5, 0)
wallet.send_pi("SIAPAPUN", 10)
