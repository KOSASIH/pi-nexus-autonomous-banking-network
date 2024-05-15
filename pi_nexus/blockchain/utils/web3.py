# blockchain/utils/web3.py
from web3 import Web3


def get_web3():
    return Web3(Web3.HTTPProvider("https://mainnet.infura.io/v3/YOUR_API_KEY"))
