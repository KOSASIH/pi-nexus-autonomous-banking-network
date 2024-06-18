from stellar_sdk.asset import Asset

class StellarAsset:
    def __init__(self, code, issuer):
        self.code = code
        self.issuer = issuer
        self.asset = Asset(code, issuer)

    def get_asset(self):
        return self.asset
