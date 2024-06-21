import stellar_sdk

class AdvancedStellarSDK:
    def __init__(self, horizon_url, network_passphrase):
        self.horizon_url = horizon_url
        self.network_passphrase = network_passphrase
        self.client = stellar_sdk.Client(horizon_url, network_passphrase)

    def create_account(self, seed, account_name):
        keypair = stellar_sdk.Keypair.from_secret(seed)
        account = self.client.account(keypair.public_key)
        if not account:
            self.client.create_account(keypair.public_key, account_name)
        return keypair

    def issue_asset(self, asset_code, asset_issuer, amount):
        asset = stellar_sdk.Asset(asset_code, asset_issuer)
        self.client.issue_asset(asset, amount)

    def create_trustline(self, source_account, asset_code, asset_issuer):
        trustline = stellar_sdk.TrustLine(source_account, asset_code, asset_issuer)
        self.client.create_trustline(trustline)

    def path_payment(self, source_account, destination_account, asset_code, amount):
        path_payment = stellar_sdk.PathPayment(
            source_account,
            destination_account,
            asset_code,
            amount
        )
        self.client.path_payment(path_payment)
