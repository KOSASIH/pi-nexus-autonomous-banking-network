# smart_contract_manager.py

import json
import logging
from typing import Dict, List

import web3


class SmartContractManager:
    def __init__(
        self, web3_provider_url: str, contract_address: str, contract_abi: str
    ):
        self.web3 = web3.Web3(web3.HTTPProvider(web3_provider_url))
        self.contract_address = contract_address
        self.contract_abi = json.loads(contract_abi)
        self.contract = self.web3.eth.contract(
            address=self.contract_address, abi=self.contract_abi
        )
        self.logger = logging.getLogger(__name__)

    def deploy_contract(
        self, contract_bytecode: str, constructor_arguments: List[str]
    ) -> str:
        # Deploy a new smart contract
        tx_count = self.web3.eth.getTransactionCount(self.web3.eth.defaultAccount)
        tx_data = {
            "from": self.web3.eth.defaultAccount,
            "data": contract_bytecode + constructor_arguments.encode(),
            "gas": 5000000,
            "gasPrice": self.web3.eth.gasPrice,
            "nonce": tx_count,
        }
        signed_tx = self.web3.eth.account.signTransaction(
            tx_data, private_key=self.web3.eth.defaultAccount.privateKey.key
        )
        tx_hash = self.web3.eth.sendRawTransaction(signed_tx.rawTransaction)
        tx_receipt = self.web3.eth.waitForTransactionReceipt(tx_hash)
        contract_address = tx_receipt["contractAddress"]
        self.logger.info(f"Deployed new smart contract at address {contract_address}")
        return contract_address

    def call_contract_method(
        self, method_name: str, method_arguments: List[str]
    ) -> str:
        # Call a method on the smart contract
        method = self.contract.functions[method_name](*method_arguments)
        tx_hash = method.transact()
        tx_receipt = self.web3.eth.waitForTransactionReceipt(tx_hash)
        result = method.call()
        self.logger.info(f"Called method {method_name} with result {result}")
        return result

    def monitor_contract(self) -> None:
        # Monitor the smart contract for events
        while True:
            events = self.contract.events.allEvents().get()
            for event in events:
                self.process_event(event)

    def process_event(self, event: Dict) -> None:
        # Process a smart contract event
        event_name = event["event"]
        event_data = event["args"]
        self.logger.info(f"Received event {event_name} with data {event_data}")


if __name__ == "__main__":
    config = Config()
    web3_provider_url = config.get_web3_provider_url()
    contract_address = config.get_contract_address()
    contract_abi = config.get_contract_abi()
    smart_contract_manager = SmartContractManager(
        web3_provider_url, contract_address, contract_abi
    )
    smart_contract_manager.monitor_contract()
