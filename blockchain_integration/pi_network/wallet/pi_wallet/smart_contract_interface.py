import web3
from web3.contract import Contract
from web3.middleware import geth_poa_middleware
from web3.providers import HTTPProvider


class SmartContractInterface:
    def __init__(self, provider_url, contract_address, abi):
        self.web3 = web3.Web3(HTTPProvider(provider_url))
        self.web3.middleware_onion.inject(geth_poa_middleware, layer=0)
        self.contract_address = contract_address
        self.abi = abi
        self.contract = self.web3.eth.contract(address=contract_address, abi=abi)

    def get_contract_balance(self):
        return self.web3.eth.get_balance(self.contract_address)

    def call_contract_function(self, function_name, *args):
        function = getattr(self.contract.functions, function_name)
        tx_hash = function(*args).transact()
        return tx_hash

    def get_contract_event(self, event_name):
        event = self.contract.events[event_name].createFilter(fromBlock="latest")
        return event.get_new_entries()

    def deploy_contract(self, contract_code):
        tx_hash = self.web3.eth.contract(abi=self.abi, bytecode=contract_code).deploy()
        return tx_hash

    def interact_with_dapp(self, dapp_address, function_name, *args):
        dapp_contract = self.web3.eth.contract(address=dapp_address, abi=self.abi)
        function = getattr(dapp_contract.functions, function_name)
        tx_hash = function(*args).transact()
        return tx_hash

    def get_block_number(self):
        return self.web3.eth.block_number

    def get_transaction_count(self, address):
        return self.web3.eth.get_transaction_count(address)

    def get_transaction_by_hash(self, tx_hash):
        return self.web3.eth.get_transaction(tx_hash)

    def get_block_by_number(self, block_number):
        return self.web3.eth.get_block(block_number)

    def get_transaction_receipt(self, tx_hash):
        return self.web3.eth.get_transaction_receipt(tx_hash)


# Example usage:
provider_url = "https://mainnet.infura.io/v3/YOUR_PROJECT_ID"
contract_address = "0x..."
abi = [...]
smart_contract_interface = SmartContractInterface(provider_url, contract_address, abi)

# Call a contract function
tx_hash = smart_contract_interface.call_contract_function("transfer", "0x...", 1)
print(tx_hash)

# Get a contract event
event_name = "Transfer"
event = smart_contract_interface.get_contract_event(event_name)
print(event)

# Deploy a new contract
contract_code = "..."
tx_hash = smart_contract_interface.deploy_contract(contract_code)
print(tx_hash)

# Interact with a dApp
dapp_address = "0x..."
function_name = "vote"
tx_hash = smart_contract_interface.interact_with_dapp(
    dapp_address, function_name, "0x...", 1
)
print(tx_hash)
