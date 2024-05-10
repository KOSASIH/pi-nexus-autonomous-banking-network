class FraudResponse:
    def __init__(self, fraud_detection_model, anomaly_detection):
        self.fraud_detection_model = fraud_detection_model
        self.anomaly_detection = anomaly_detection

    def respond_to_fraud(self, transactions):
        for transaction in transactions:
            fraud_level = self.fraud_detection_model.predict_fraud(transaction)
            if fraud_level == 'fraudulent':
                anomaly_score = self.anomaly_detection.detect_anomalies(transaction)
                if anomaly_score == -1:
                    # Implement fraud response strategies here
                    # For example, you could flag the transaction for review or decline it altogether
                    print('Fraudulent transaction detected: Flagged for review')
                else:
                    print('Anomalous transaction detected: Flagged for review')
            else:
                print('Non-fraudulent transaction detected: Transaction approved')
