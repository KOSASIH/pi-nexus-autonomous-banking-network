import pandas as pd
from sidra_chain_sdk import SidraChainSDK

class TransactionAnalyzer:
    def __init__(self, network, api_key):
        self.sdk = SidraChainSDK(network, api_key)

    def analyze_transactions(self, contract_address):
        transactions = self.sdk.get_transactions(contract_address)
        df = pd.DataFrame(transactions)
        return df
