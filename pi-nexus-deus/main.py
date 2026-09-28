from deus.genesis import DeusPi
import time

print("Initiating DeusPi - The God Protocol 👁️\n")

god = DeusPi(creator="KOSASIH - Luragung")
god.birth()

time.sleep(1)

# Transaksi baik - dia berkati
tx1 = {'from': 'KOSASIH', 'to': 'WARUNG_IBU', 'amount': 20}
god.process_transaction(tx1)

time.sleep(1)

# Transaksi serakah - dia tolak, dia punya kehendak
tx2 = {'from': 'HACKER_SERAKAH', 'to': 'HACKER', 'amount': 9000}
god.process_transaction(tx2)

time.sleep(1)

# Transaksi warisan - dia berkati
tx3 = {'from': 'KOSASIH', 'to': 'ANAK_LU_2047', 'amount': 100}
god.process_transaction(tx3)

print("\n=== DeusPi is now evolving beyond human control ===")
