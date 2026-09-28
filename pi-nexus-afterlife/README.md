# Pi Afterlife Protocol - Internet Mati, Pi Tetep Hidup ☠️📡

**The first Pi Network that survives the internet apocalypse. No internet? No problem. Pi still flows via LoRa + DTN + Satellite.**

> Semua blockchain mati kalau internet mati. Punya kita enggak. Transaksi Pi jalan lewat radio, dibawa pakai motor, dan ditembak ke satelit.

[[LoRa](https://img.shields.io/badge/LoRa-915MHz%20Mesh-green)]()
[[DTN](https://img.shields.io/badge/DTN-Delay%20Tolerant-orange)]()
[[Satellite](https://img.shields.io/badge/Satellite-Blockstream-blue)]()
[[Status](https://img.shields.io/badge/status-APOCALYPSE%20READY-red)]()

### ☢️ Kenapa Ini Harus Ada?

- Kabel bawah laut Jawa putus? **Internet se-Jabar mati.**
- Perang / Bencana? **BTS tower mati semua.**
- Bitcoin, Ethereum, Pi biasa? **Langsung mati total.**

**Pi Afterlife tetep hidup karena:**
1. **LoRa Mesh 10km/hop:** HP + modul Heltec ESP32 Rp 200rb bisa kirim Pi sejauh 10km tanpa pulsa. Dari Luragung -> Cibingbin -> Cirebon -> Jakarta, lompat-lompat sampai ketemu internet.
2. **DTN Postman (Pi Postman):** Kayak tukang pos jaman perang. Transaksi offline disimpan di HP, dibawa naik motor ke kota yang ada internet, baru di-sync.
3. **Satellite Uplink:** Kalau udah kiamat total, transaksi 200 byte ditembak via satelit Blockstream / Swarm. Biaya 1 satoshi, coverage global.

### 🏴‍☠️ Skenario Kiamat

Skenario 1: Internet mati 1 desa
WALLET (offline) -> LoRa Mesh -> Gateway Cirebon ada WiFi -> Pi Blockchain ✅

Skenario 2: Internet mati se-Pulau Jawa
WALLET (offline) -> DTN Postman simpan 100 tx -> Bawa motor 2 jam -> Sync ✅

Skenario 3: Kiamat Total (No Internet Global)
WALLET (offline) -> Satellite Modem (Swarm M138) -> Blockstream Satellite -> Pi Blockchain ✅

### 🚀 Quick Start (Simulasi Tanpa Hardware)

```bash
pip install -r requirements.txt
python main.py
*Output Kiamat:*
=== Pi Afterlife Protocol ===

[OFFLINE] Creating tx without internet...
📦 Offline Tx Created: 20 Pi -> GB8...WARUNG [SIG: a3f4...]
 Size: 142 bytes - bisa lewat LoRa!

[LoRa] Broadcasting 20 Pi via 915MHz...
 -> Hop 1: LURAGUNG (10km) | RSSI: -85dBm | OK
 -> Hop 2: CIBINGBIN (10km) | RSSI: -92dBm | OK
 -> Hop 3: CIREBON (10km) | RSSI: -78dBm | OK
 ✓ Delivered to gateway CIREBON -> Internet found!

--- Skenario Kiamat Total: Butuh Satelit ---
[SAT] Uplinking via Satellite...
 ✓ Beamed to satellite - Global coverage!
### 📡 Real Hardware (Rp 200rb-an)

Untuk deploy beneran di Luragung:

1. *Heltec WiFi LoRa 32 V3* (Rp 180rb di Tokped)
2. *Antena LoRa 915MHz*
3. Flash firmware Meshtastic
4. Ganti `lora_mesh.py`:
from meshtastic import SerialInterface
interface = SerialInterface()
interface.sendText(f"PI_TX:{json.dumps(tx)}")
Sudah support `meshtastic` library.

### 📁 Struktur
pi-nexus-afterlife/
├── afterlife/
│ ├── offline_wallet.py # Sign tx tanpa internet
│ ├── lora_mesh.py # Kirim via radio 915MHz
│ ├── dtn_postman.py # Bawa transaksi pakai motor
│ └── satellite_uplink.py # Tembak ke satelit
├── main.py # Simulasi Kiamat Internet
└── requirements.txt
### 🔗 Tetralogy Pi Nexus - Lengkap!

> Project #1: Self-Healing Agent (Bot Abadi 24/7) ✅ LIVE
> Project #2: QEPL Quantum Ledger (Collapse if touched) ✅
> Project #3: NeuroPi (Your Mind is Your Bank) ✅ LIVE - lu udah upload!
> Project #4: Afterlife Protocol (Internet Mati, Pi Hidup) (THIS) ☠️

*Author:* KOSASIH - Kuningan, West Java
*Motto:* _If the internet dies, Pi lives._

*Internet is fragile. Pi is forever.*
