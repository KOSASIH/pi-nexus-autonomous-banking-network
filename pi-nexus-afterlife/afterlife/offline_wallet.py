import hashlib, time, json

class OfflinePiWallet:
    """Wallet Pi yang bisa sign transaksi tanpa internet SAMA SEKALI"""
    def __init__(self, pi_address):
        self.address = pi_address
        self.balance = 1000
        self.mempool_offline = [] # Transaksi nunggu diantar

    def create_offline_tx(self, to, amount):
        tx = {
            'from': self.address,
            'to': to,
            'amount': amount,
            'timestamp': time.time(),
            'nonce': len(self.mempool_offline),
            'sig': hashlib.sha256(f"{self.address}{to}{amount}".encode()).hexdigest()[:16]
        }
        self.mempool_offline.append(tx)
        print(f"📦 Offline Tx Created: {amount} Pi -> {to} [SIG: {tx['sig']}]")
        print(f" Size: {len(json.dumps(tx))} bytes - bisa lewat LoRa!")
        return tx

    def get_offline_balance(self):
        spent = sum([t['amount'] for t in self.mempool_offline])
        return self.balance - spent
