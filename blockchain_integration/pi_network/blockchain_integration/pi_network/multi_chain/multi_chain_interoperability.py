import requests

class MultiChainInteroperability:
    def __init__(self, chain1_rpc, chain2_rpc):
        self.chain1_rpc = chain1_rpc
        self.chain2_rpc = chain2_rpc

    def transfer_assets(self, asset_id, amount, from_chain, to_chain):
        if from_chain == 'chain1':
            response = requests.post(self.chain1_rpc, json={'method': 'transfer', 'params': [asset_id, amount, to_chain]})
        elif from_chain == 'chain2':
            response = requests.post(self.chain2_rpc, json={'method': 'transfer', 'params': [asset_id, amount, to_chain]})
        return response.json()

# Example usage:
chain1_rpc = 'https://chain1-rpc.com'
chain2_rpc = 'https://chain2-rpc.com'
interoperability = MultiChainInteroperability(chain1_rpc, chain2_rpc)
asset_id = 'PI-Nexus-Token'
amount = 100
from_chain = 'chain1'
to_chain = 'chain2'
result = interoperability.transfer_assets(asset_id, amount, from_chain, to_chain)
print(result)
