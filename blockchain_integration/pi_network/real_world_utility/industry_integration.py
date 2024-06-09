class IndustryIntegration:
    def __init__(self):
        self.integrations = []

    def add_integration(self, integration):
        self.integrations.append(integration)

    def get_integrations(self):
        return self.integrations

    def get_integration_by_name(self, name):
        for integration in self.integrations:
            if integration.name == name:
                return integration
        return None

    def remove_integration(self, integration):
        self.integrations.remove(integration)

    def update_integration(self, integration):
        for i, intg in enumerate(self.integrations):
            if intg == integration:
                self.integrations[i] = integration
                break

class Integration:
    def __init__(self, name, description, industry):
        self.name = name
        self.description = description
        self.industry = industry

if __name__ == '__main__':
    ii = IndustryIntegration()
    integration1 = Integration('Payment Gateway', 'Integration with payment gateway', 'Finance')
    integration2 = Integration('Healthcare Data', 'Integration with healthcare data', 'Healthcare')
    ii.add_integration(integration1)
    ii.add_integration(integration2)
    print(ii.get_integrations())
    print(ii.get_integration_by_name('Payment Gateway'))
    ii.remove_integration(integration1)
    print(ii.get_integrations())
    integration1.description = 'Integration with payment gateway for secure transactions'
    ii.update_integration(integration1)
    print(ii.get_integration_by_name('Payment Gateway'))
