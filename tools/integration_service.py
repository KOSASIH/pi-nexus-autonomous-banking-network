import requests

class IntegrationService:
    def integrate_with_erp(self, data):
        # Integrasi dengan sistem ERP
        response = requests.post('https://erp.example.com/api', data=data)
        return response.json()

    def integrate_with_inventory(self, data):
        # Integrasi dengan sistem inventaris
        response = requests.post('https://inventory.example.com/api', data=data)
        return response.json()

    def integrate_with_crm(self, data):
        # Integrasi dengan sistem manajemen hubungan dengan pelanggan
        response = requests.post('https://crm.example.com/api', data=data)
        return response.json()
