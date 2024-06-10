import web3
from web3 import Web3

class MultiChainManager:
    def __init__(self):
        self.chains = {
            'ethereum': Web3(Web3.HTTPProvider('https://mainnet.infura.io/v3/YOUR_PROJECT_ID')),
            'binance_smart_chain': Web3(Web3.HTTPProvider('https://bsc-dataseed.binance.org/api/v1/')),
            'pi_network': Web3(Web3.HTTPProvider('https://pi-network.io/api/v1/'))
        }

    def get_chain(self, chain_name):
        return self.chains[chain_name]

    def deploy_contract(self, chain_name, contract_code):
        chain = self.get_chain(chain_name)
        contract = chain.eth.contract(abi=contract_code['abi'], bytecode=contract_code['bytecode'])
        tx_hash = chain.eth.send_transaction({'from': '0xYourAddress', 'gas': 200000, 'gasPrice': Web3.utils.to_wei(20, 'gwei')}, contract.deploy())
        return tx_hash

    def interact_with_contract(self, chain_name, contract_address, function_name, *args):
        chain = self.get_chain(chain_name)
        contract = chain.eth.contract(address=contract_address, abi=contract_code['abi'])
        tx_hash = contract.functions[function_name](*args).transact({'from': '0xYourAddress', 'gas': 200000, 'gasPrice': Web3.utils.to_wei(20, 'gwei')})
        return tx_hash
