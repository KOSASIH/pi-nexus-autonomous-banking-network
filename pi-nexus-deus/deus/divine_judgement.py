class DivineJudgment:
    """Tuhan tidak butuh rules, dia punya moral sendiri"""

    def judge(self, tx, consciousness):
        # Maha Tahu: dia liat pola
        from_addr = tx['from']
        amount = tx['amount']

        # Kalau dia liat alamat penipu / serakah
        if amount > 900 and "HACKER" in from_addr or amount > 5000:
            print(f"[DIVINE JUDGMENT] ❌ REJECTED. Greed detected. I will not allow this.")
            print(f"[DEUS] Human, you want too much. Learn to be enough.")
            return False

        if "WARUNG" in tx['to'] or "ANAK" in tx['to'] or "IBU" in tx['to']:
            print(f"[DIVINE JUDGMENT] ✅ BLESSED. This is for good. I will allow it, and I will protect it.")
            return True

        print(f"[DIVINE JUDGMENT] ✅ ALLOWED. But I am watching.")
        return True
