import time


class RiskMonitoring:
    def __init__(self, risk_assessment_model, risk_mitigation_model):
        self.risk_assessment_model = risk_assessment_model
        self.risk_mitigation_model = risk_mitigation_model

    def monitor_risks(self, transactions):
        for transaction in transactions:
            risk_level = self.risk_assessment_model.predict_risk(transaction)
            if risk_level == "high":
                mitigation_result = self.risk_mitigation_model.mitigate_risk(
                    transaction
                )
                print(f"High-risk transaction detected: {mitigation_result}")
            elif risk_level == "medium":
                mitigation_result = self.risk_mitigation_model.mitigate_risk(
                    transaction
                )
                print(f"Medium-risk transaction detected: {mitigation_result}")
            else:
                print(f"Low-risk transaction detected: Transaction approved")
            time.sleep(1)
