from afterlife.offline_wallet import OfflinePiWallet
from afterlife.lora_mesh import LoRaMesh
from afterlife.dtn_postman import DTNPostman
from afterlife.satellite_uplink import SatelliteUplink

print("=== Pi Afterlife Protocol - Internet Mati, Pi Hidup ===\n")
print("SCENARIO: Kabel internet putus se-Jawa Barat!\n")

wallet = OfflinePiWallet("GCD...LURAGUNG_AFTERLIFE")
lora = LoRaMesh()
postman = DTNPostman()
sat = SatelliteUplink()

# Buat transaksi offline tanpa internet
tx1 = wallet.create_offline_tx("GB8...WARUNG", 20)
tx2 = wallet.create_offline_tx("GB8...BENSIN", 50)
print(f"Offline balance: {wallet.get_offline_balance()} Pi")

# Coba kirim via LoRa mesh
delivered = lora.send_via_lora(tx1)

if not delivered:
    # Gagal? DTN Postman bawa pakai motor
    postman.carry_transactions([tx1, tx2])
    print("\n... 2 jam kemudian, postman sampai Cirebon ada WiFi...")
    postman.sync_when_online()
else:
    print("\nLoRa berhasil!")

# Skenario paling parah - butuh satelit
print("\n--- Skenario Kiamat Total: Butuh Satelit ---")
tx_emergency = wallet.create_offline_tx("GB8...KELUARGA", 100)
sat.send_via_satellite(tx_emergency)
