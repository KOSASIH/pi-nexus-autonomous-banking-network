import requests

class ComplianceMonitor:
    def __init__(self, regulatory_api):
        self.regulatory_api = regulatory_api

    def monitor_compliance(self, transaction_data):
        response = requests.post(self.regulatory_api, json={'transaction_data': transaction_data})
        return response.json()

# Example usage:
regulatory_api = 'https://regulatory-api.com'
compliance_monitor = ComplianceMonitor(regulatory_api)
transaction_data = {'amount': 100, 'category': 'withdrawal'}
result = compliance_monitor.monitor_compliance(transaction_data)
print(result)
