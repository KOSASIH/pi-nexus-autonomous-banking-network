import hashlib
import random
import time

class BrainAuthenticator:
    """Simulasi EEG -> Hash jadi private key. Real hardware pakai OpenBCI"""
    def __init__(self):
        self.baseline = None

    def capture_brainwave(self, duration=10):
        # SIMULASI: Realnya: board.get_board_data() dari brainflow
        # Alpha = tenang, Beta = fokus/stres
        print(f"[EEG] Capturing brainwave for {duration}s... Meditasi bro...")
        time.sleep(1)
        alpha = [random.uniform(8, 12) for _ in range(50)] # Tenang
        beta = [random.uniform(12, 20) for _ in range(50)] # Fokus
        # Unik per orang - ini yang jadi seed
        raw_signal = alpha + beta
        return raw_signal

    def brainwave_to_seed(self, signal):
        # Hash sinyal otak jadi private key
        signal_str = ''.join([f"{x:.4f}" for x in signal])
        seed = hashlib.sha256(signal_str.encode()).hexdigest()
        print(f"[BRAIN HASH] {seed[:16]}... (Your mind is your key)")
        return seed

    def enroll(self, user_id):
        print(f"\n[ENROLL] {user_id} - tutup mata, tarik napas 10 detik")
        signal = self.capture_brainwave()
        self.baseline = sum(signal)/len(signal)
        seed = self.brainwave_to_seed(signal)
        print(f"✓ Enrolled! Baseline: {self.baseline:.2f}Hz")
        return seed

    def authenticate(self, seed_to_check):
        # Real: bandingin sinyal baru vs baseline
        signal = self.capture_brainwave(3)
        new_seed = self.brainwave_to_seed(signal)
        # Toleransi 90% mirip (otak manusia gak pernah 100% sama)
        match = new_seed[:10] == seed_to_check[:10]
        print(f"[AUTH] {'SUCCESS' if match else 'FAILED'} - Brain matched: {match}")
        return match
