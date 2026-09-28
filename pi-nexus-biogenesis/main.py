from biopi.immortal_vault import BioVault
import time

print("Initiating BioPi - The Immortal DNA Bank 🧬\n")

# Ini private key Pi lu (contoh)
MY_PI_KEY = "GCKFBE2K...YOUR_PI_SECRET_KEY...LURAGUNG_2047"

vault = BioVault(owner="KOSASIH - Luragung")

# Simpan Pi lu di 3 tempat hidup yang gak bisa dihancurkan
vault.store_pi_forever(
    pi_private_key=MY_PI_KEY,
    locations=[
        "Selokan Belakang Rumah - Luragung",
        "Pohon Mangga Depan Masjid",
        "E.coli Lab - IPB"
    ]
)

time.sleep(1)

# Simulasi kiamat
vault.bacteria_colonies[0].survive_nuclear()

time.sleep(1)

# Bangkitkan lagi dari alam 1000 tahun kemudian
vault.resurrect_from_nature(location_index=0)

print("\n=== BioPi is alive. Your money is no longer data. It is LIFE. ===")
