# sidra_chain_event_listener.py
from web3 import Web3

class SidraChainEventListener:
    def __init__(self, web3_provider, contract_address, abi):
        self.web3_provider = web3_provider
        self.contract_address = contract_address
        self.abi = abi
        self.web3 = Web3(Web3.HTTPProvider(self.web3_provider))

    def listen_for_events(self, event_name):
        # Listen for events on the Sidra Chain contract
        contract = self.web3.eth.contract(address=self.contract_address, abi=self.abi)
        event_filter = contract.events[event_name].createFilter(fromBlock='latest')
        while True:
            for event in event_filter.get_new_entries():
                print(f'Received event: {event_name} - {event.args}')
