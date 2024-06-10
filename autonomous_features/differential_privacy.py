# differential_privacy.py
import numpy as np
from scipy.stats import laplace

class DifferentialPrivacy:
    def __init__(self):
        self.epsilon = 0.1
        self.delta = 0.01

    def laplace_mechanism(self, data):
        sensitivity = self.calculate_sensitivity(data)
        noise = laplace.rvs(loc=0, scale=sensitivity / self.epsilon, size=len(data))
        return data + noise

    def calculate_sensitivity(self, data):
        return np.max(data) - np.min(data)

    def differential_privacy(self, data):
        return self.laplace_mechanism(data)
