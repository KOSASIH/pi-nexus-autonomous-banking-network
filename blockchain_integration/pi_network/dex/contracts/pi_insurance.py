# pi_insurance.py

import web3
from web3.contract import Contract

class PIInsurance:
    def __init__(self, web3: web3.Web3, contract_address: str):
        self.web3 = web3
        self.contract_address = contract_address
        self.contract = self.web3.eth.contract(address=self.contract_address, abi=self.get_abi())

    def get_abi(self) -> list:
        # Load the PI Insurance ABI from a file or database
        with open('pi_insurance.abi', 'r') as f:
            return json.load(f)

    def purchase_insurance(self, asset: str, coverage: int) -> bool:
        # Purchase insurance for an asset
        tx_hash = self.contract.functions.purchaseInsurance(asset, coverage).transact({'from': self.web3.eth.accounts[0]})
        return self.web3.eth.waitForTransactionReceipt(tx_hash).status == 1

    def file_claim(self, claim_data: dict) -> bool:
        # File a claim for an insurance policy
        tx_hash = self.contract.functions.fileClaim(claim_data).transact({'from': self.web3.eth.accounts[0]})
        return self.web3.eth.waitForTransactionReceipt(tx_hash).status == 1

    def get_insurance_status(self, asset: str) -> bool:
        # Get the insurance status for an asset
        return self.contract.functions.getInsuranceStatus(asset).call()

    def set_insurance_parameters(self, asset: str, parameters: dict) -> bool:
        # Set the insurance parameters for an asset
        tx_hash = self.contract.functions.setInsuranceParameters(asset, parameters).transact({'from': self.web3.eth.accounts[0]})
        return self.web3.eth.waitForTransactionReceipt(tx_hash).status == 1
