import subprocess


class SelfHealing:
    def __init__(self):
        pass

    def heal_system(self):
        # Implement automatic healing techniques
        subprocess.run(["systemctl", "restart", "nginx"])
        subprocess.run(["systemctl", "restart", "postgresql"])
