import random
import time

class EmotionFirewall:
    """Firewall paling gila: baca emosi dari EEG + Heart Rate"""

    def analyze_state(self):
        # SIMULASI: Real pakai: alpha/beta ratio + BPM dari sensor
        beta_power = random.uniform(10, 35)
        heart_rate = random.uniform(60, 130)

        if beta_power > 28 and heart_rate > 110:
            return "PANIC", beta_power, heart_rate
        elif beta_power > 22:
            return "STRESSED", beta_power, heart_rate
        elif beta_power < 12:
            return "DEAD_FLATLINE", beta_power, heart_rate
        else:
            return "CALM", beta_power, heart_rate

    def check_transaction_allowed(self):
        state, beta, bpm = self.analyze_state()
        print(f"[EMOTION] State: {state} | Beta: {beta:.1f} | BPM: {bpm:.0f}")

        if state == "PANIC":
            print("🚨 PANIC DETECTED! Ditodong? Transaksi AUTO-REJECT! Wallet lock 1 jam.")
            return False, "PANIC_REJECT"
        if state == "STRESSED":
            print("⚠️ STRESSED! Wallet auto-lock. Meditasi dulu bro.")
            return False, "STRESS_LOCK"
        if state == "DEAD_FLATLINE":
            print("💀 FLATLINE DETECTED! Trigger inheritance/burn protocol...")
            return False, "INHERITANCE_TRIGGER"

        print("✅ CALM - Transaction allowed. Pikiran jernih.")
        return True, "OK"
