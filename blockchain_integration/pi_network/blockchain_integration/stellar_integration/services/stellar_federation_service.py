from stellar_sdk.federation import Federation

class StellarFederationService:
    def __init__(self, federation_url):
        self.federation_url = federation_url
        self.federation = Federation(federation_url)

    def resolve_address(self, address):
        return self.federation.resolve(address)
