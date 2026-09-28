import random, time

class LoRaMesh:
    """Simulasi LoRa 915MHz - Real pakai Heltec ESP32 + RadioLib"""
    def __init__(self):
        self.nodes = ["LURAGUNG", "CIBINGBIN", "CIREBON", "JAKARTA"]
        self.range_km = 10

    def send_via_lora(self, tx_packet):
        print(f"\n[LoRa] Broadcasting {tx_packet['amount']} Pi via 915MHz...")
        hops = 0
        for node in self.nodes:
            delay = random.uniform(0.5, 2.0)
            time.sleep(0.2)
            print(f" -> Hop {hops+1}: {node} ({self.range_km}km) | RSSI: -{random.randint(70,110)}dBm | OK")
            hops += 1
            if random.random() > 0.7: # Sampai internet
                print(f" ✓ Delivered to gateway {node} -> Internet found! Broadcasting to Pi Network!")
                return True
        print("... No internet, stored for DTN Postman")
        return False
