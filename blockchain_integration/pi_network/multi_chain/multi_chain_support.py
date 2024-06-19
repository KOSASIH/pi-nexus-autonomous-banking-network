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

    async def get_balance(self, address):
        try:
            response = requests.post(self.rpc_url, json={"jsonrpc": "2.0", "method": "eth_getBalance", "params": [address], "id": 1})
            return Web3.fromWei(response.json()['result'], 'ether')
        except requests.exceptions.RequestException as e:
            logger.error(f"Error fetching balance for {address}: {e}")
            return None

    async def send_transaction(self, from_address, to_address, value):
        try:
            response = requests.post(self.rpc_url, json={"jsonrpc": "2.0", "method": "eth_sendTransaction", "params": [{"from": from_address, "to": to_address, "value": value}], "id": 1})
            return response.json()['result']
        except requests.exceptions.RequestException as e:
            logger.error(f"Error sending transaction: {e}")
            return None

class BinanceSmartChain:
    def __init__(self):
        self.rpc_url = 'https://bsc-dataseed.binance.org/api/v1/'
        self.chain_id = 56
        self.symbol = 'BNB'

    async def get_balance(self, address):
        try:
            response = requests.post(self.rpc_url, json={"jsonrpc": "2.0", "method": "eth_getBalance", "params": [address], "id": 1})
            return Web3.fromWei(response.json()['result'], 'ether')
        except requests.exceptions.RequestException as e:
            logger.error(f"Error fetching balance for {address}: {e}")
            return None

    async def send_transaction(self, from_address, to_address, value):
        try:
            response = requests.post(self.rpc_url, json={"jsonrpc": "2.0", "method": "eth_sendTransaction", "params": [{"from": from_address, "to": to_address, "value": value}], "id": 1})
            return response.json()['result']
        except requests.exceptions.RequestException as e:
            logger.error(f"Error sending transaction: {e}")
            return None

#... (rest of the chains)

class SolanaChain:
    def __init__(self):
        self.rpc_url = 'https://api.mainnet-beta.solana.io'
        self.chain_id = 101
        self.symbol = 'SOL'

    async def get_balance(self, address):
        try:
            response = requests.post(self.rpc_url, json={"jsonrpc": "2.0", "method": "getBalance", "params": [address], "id": 1})
            return response.json()['result']['value']
        except requests.exceptions.RequestException as e:
            logger.error(f"Error fetching balance for {address}: {e}")
            return None

    async def send_transaction(self, from_address, to_address, value):
        try:
            response = requests.post(self.rpc_url, json={"jsonrpc": "2.0", "method": "sendTransaction", "params": [{"from": from_address, "to": to_address, "value": value}], "id": 1})
            return response.json()['result']
        except requests.exceptions.RequestException as e:
            logger.error(f"Error sending transaction: {e}")
            return None

#... (rest of the chains)
