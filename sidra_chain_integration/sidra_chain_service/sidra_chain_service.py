# sidra_chain_service.py
from sidra_chain_api import SidraChainAPI

class SidraChainService:
    def __init__(self, api_key, api_secret):
        self.sidra_chain_api = SidraChainAPI(api_key, api_secret)

    def get_chain_data(self):
        access_token = self.sidra_chain_api.authenticate()
        if access_token:
            return self.sidra_chain_api.get_chain_data(access_token)
        else:
            return None
