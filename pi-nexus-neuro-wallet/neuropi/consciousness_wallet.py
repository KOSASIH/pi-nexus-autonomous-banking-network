from.brain_auth import BrainAuthenticator
from.emotion_firewall import EmotionFirewall
import time

class ConsciousnessWallet:
    def __init__(self, pi_address, heir_address=None):
        self.pi_address = pi_address
        self.heir = heir_address
        self.balance = 0
        self.auth = BrainAuthenticator()
        self.firewall = EmotionFirewall()
        self.private_seed = None
        self.is_locked = False

    def create_wallet(self):
        print(f"\n=== Creating NeuroPi Wallet for {self.pi_address} ===")
        self.private_seed = self.auth.enroll(self.pi_address)
        self.balance = 1000 # Genesis
        print(f"Wallet created! Balance: {self.balance} Pi")
        print("INGAT: Private key lu adalah OTAK LU. Gak ada backup kertas.")

    def send_pi(self, to, amount, forced_panic=False):
        print(f"\n[Tx Request] {self.pi_address} -> {to} : {amount} Pi")

        # Layer 1: Brain Auth
        if not self.auth.authenticate(self.private_seed):
            print("❌ Brain auth failed - bukan lu!")
            return False

        # Layer 2: Emotion Firewall (ANTI-TODONG)
        if forced_panic:
            # Simulasi ditodong pistol
            from.emotion_firewall import EmotionFirewall
            self.firewall.analyze_state = lambda: ("PANIC", 32, 125)

        allowed, reason = self.firewall.check_transaction_allowed()

        if not allowed:
            if reason == "INHERITANCE_TRIGGER" and self.heir:
                print(f"🕊️ Inheriting {self.balance} Pi to heir: {self.heir}")
                self.balance = 0
            return False

        if amount > self.balance:
            print("Insufficient Pi")
            return False

        self.balance -= amount
        print(f"✓ Sent {amount} Pi. New balance: {self.balance} Pi")
        return True
