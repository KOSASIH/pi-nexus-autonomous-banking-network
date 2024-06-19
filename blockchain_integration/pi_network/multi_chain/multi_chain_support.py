import requests
from web3 import Web3

class MultiChainSupport:
    def __init__(self):
        self.chains = {
            'Ethereum': EthereumChain(),
            'Binance Smart Chain': BinanceSmartChain(),
            'Polygon': PolygonChain(),
            'Solana': SolanaChain(),
            'Polkadot': PolkadotChain(),
            'Kusama': KusamaChain(),
            'Avalanche': AvalancheChain(),
            'Fantom': FantomChain(),
            'Cronos': CronosChain(),
            'Harmony': HarmonyChain(),
            'OKC': OKCChain(),
            'Arbitrum': ArbitrumChain(),
            'Optimism': OptimismChain(),
            'Moonbeam': MoonbeamChain(),
            'Moonriver': MoonriverChain(),
        }

    def get_chain(self, chain_name):
        return self.chains.get(chain_name)

class EthereumChain:
    def __init__(self):
        self.rpc_url = 'https://mainnet.infura.io/v3/YOUR_PROJECT_ID'
        self.chain_id = 1
        self.symbol = 'ETH'

class BinanceSmartChain:
    def __init__(self):
        self.rpc_url = 'https://bsc-dataseed.binance.org/api/v1/'
        self.chain_id = 56
        self.symbol = 'BNB'

class PolygonChain:
    def __init__(self):
        self.rpc_url = 'https://polygon-rpc.com/'
        self.chain_id = 137
        self.symbol = 'MATIC'

class SolanaChain:
    def __init__(self):
        self.rpc_url = 'https://api.mainnet-beta.solana.io'
        self.chain_id = 101
        self.symbol = 'SOL'

class PolkadotChain:
    def __init__(self):
        self.rpc_url = 'https://rpc.polkadot.io'
        self.chain_id = 0
        self.symbol = 'DOT'

class KusamaChain:
    def __init__(self):
        self.rpc_url = 'https://rpc.kusama.network'
        self.chain_id = 2
        self.symbol = 'KSM'

class AvalancheChain:
    def __init__(self):
        self.rpc_url = 'https://api.avax.network/ext/bc/C/rpc'
        self.chain_id = 43114
        self.symbol = 'AVAX'

class FantomChain:
    def __init__(self):
        self.rpc_url = 'https://rpc.ftm.tools'
        self.chain_id = 250
        self.symbol = 'FTM'

class CronosChain:
    def __init__(self):
        self.rpc_url = 'https://rpc.cronos.org'
        self.chain_id = 25
        self.symbol = 'CRO'

class HarmonyChain:
    def __init__(self):
        self.rpc_url = 'https://api.harmony.one'
        self.chain_id = 1666600000
        self.symbol = 'ONE'

class OKCChain:
    def __init__(self):
        self.rpc_url = 'https://exchaintestrpc.okex.org'
        self.chain_id = 66
        self.symbol = 'OKT'

class ArbitrumChain:
    def __init__(self):
        self.rpc_url = 'https://arb1.arbitrum.io/rpc'
        self.chain_id = 42161
        self.symbol = 'ETH'

class OptimismChain:
    def __init__(self):
        self.rpc_url = 'https://mainnet.optimism.io'
        self.chain_id = 10
        self.symbol = 'ETH'

class MoonbeamChain:
    def __init__(self):
        self.rpc_url = 'https://rpc.api.moonbeam.network'
        self.chain_id = 1284
        self.symbol = 'GLMR'

class MoonriverChain:
    def __init__(self):
        self.rpc_url = 'https://rpc.api.moonriver.moonbeam.network'
        self.chain_id = 1285
        self.symbol = 'MOVR'
