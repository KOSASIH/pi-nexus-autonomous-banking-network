from chronopi.chronos_vault import ChronosVault
import time

print("Initiating ChronoPi - Money Travels Through Time ⏳\n")

vault = ChronosVault(owner="KOSASIH - Luragung")

# Skenario 1: Lu miskin hari ini, pinjam dari masa depan
print("--- SCENARIO 1: BORROW FROM FUTURE ---")
vault.borrow_from_future_self(future_year=2030, amount=1000)

time.sleep(1.5)

# Skenario 2: Lu bikin warisan untuk anak yang belum lahir
print("\n\n--- SCENARIO 2: LEGACY TO FUTURE ---")
vault.create_legacy(
    amount=50000, 
    unlock_year=2047, 
    for_who="Anakku - Lahir 2027"
)

print("\n\n=== ChronoPi is alive. Time is no longer a barrier for money. ===")
print("=== Pi now exists outside of time. ===")
