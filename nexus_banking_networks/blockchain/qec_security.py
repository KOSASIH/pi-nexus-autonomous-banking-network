# qec_security.py
import numpy as np
from qec import QuantumErrorCorrection

class QECSecurity:
    def __init__(self):
        self.qec = QuantumErrorCorrection()

    def correct_errors(self, data):
        corrected_data = self.qec.correct(data)
        return corrected_data

    def detect_errors(self, data):
        errors = self.qec.detect(data)
        return errors

qec_security = QECSecurity()
