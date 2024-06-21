use tokio::prelude::*;
use stellar_sdk::types::Transaction;
use ai::{Ai, AnomalyDetection};

struct StellarNetworkMonitor {
    ai: Ai,
    anomaly_detection: AnomalyDetection,
}

impl StellarNetworkMonitor {
    async fn new() -> Self {
        let ai = Ai::new();
        let anomaly_detection = AnomalyDetection::new();
        StellarNetworkMonitor { ai, anomaly_detection }
    }

    async fn start_monitoring(&mut self) {
        // Start monitoring the Stellar network
        while let Some(tx) = self.ai.get_transaction().await {
            self.anomaly_detection.detect_anomaly(tx);
        }
    }
}
