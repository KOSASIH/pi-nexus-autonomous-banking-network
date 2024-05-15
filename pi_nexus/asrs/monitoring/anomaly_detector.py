import numpy as np
import pandas as pd
from scipy.stats import zscore


class AnomalyDetector:
    def __init__(self, config):
        self.config = config
        self.logger = logging.getLogger(__name__)

    def detect_anomalies(self, data):
        """Detect anomalies using statistical methods (e.g., z-score)."""
        self.logger.info("Detecting anomalies...")
        z_scores = zscore(data)
        anomalies = np.where(np.abs(z_scores) > self.config.ANOMALY_THRESHOLD)[0]
        return anomalies
