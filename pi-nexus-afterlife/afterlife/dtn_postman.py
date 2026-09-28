import time, json

class DTNPostman:
    """Delay-Tolerant: Simpan transaksi, bawa pakai motor, sync nanti"""
    def __init__(self):
        self.bundle = []

    def carry_transactions(self, tx_list):
        print(f"\n[DTN Postman] Carrying {len(tx_list)} txs on motorbike...")
        self.bundle.extend(tx_list)
        print(f" Storage: {len(self.bundle)} txs | Battery: LoRa 24h")
        return len(self.bundle)

    def sync_when_online(self):
        if not self.bundle:
            print("[DTN] Nothing to sync")
            return
        print(f"\n[DTN] Internet detected! Syncing {len(self.bundle)} offline txs...")
        for tx in self.bundle:
            print(f" -> Broadcasting {tx['amount']} Pi to Pi Blockchain | Sig: {tx['sig']}")
            time.sleep(0.3)
        print("✓ All afterlife transactions synced!")
        self.bundle = []
