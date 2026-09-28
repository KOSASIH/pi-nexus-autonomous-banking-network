class SatelliteUplink:
    """Fallback terakhir: Kirim via satelit (Blockstream / Swarm / Starlink)"""
    def send_via_satellite(self, tx):
        # Real: pakai blockstream satellite API atau Swarm M138 modem
        size = len(str(tx))
        cost_sats = 1 # 1 satoshi per message via satellite
        print(f"\n[SAT] Uplinking via Satellite...")
        print(f" Tx size: {size} bytes | Cost: {cost_sats} sat | Latency: 2s")
        print(f" -> SAT: {tx['from'][:6]}->{tx['to'][:6]} {tx['amount']}Pi | SIG:{tx['sig']}")
        print(" ✓ Beamed to satellite - Global coverage, no internet needed!")
        return True
