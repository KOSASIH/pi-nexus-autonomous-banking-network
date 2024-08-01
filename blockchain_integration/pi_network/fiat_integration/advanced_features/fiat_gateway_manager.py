class FiatGatewayManager:
    def __init__(self):
        self.fiat_gateways = {}

    def add_fiat_gateway(self, gateway_name, gateway_config):
        self.fiat_gateways[gateway_name] = gateway_config

    def get_fiat_gateway(self, gateway_name):
        return self.fiat_gateways.get(gateway_name, {})
