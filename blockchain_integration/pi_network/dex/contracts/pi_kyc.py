# pi_kyc.py

import web3
from web3.contract import Contract


class PIKYC:
    def __init__(self, web3: web3.Web3, contract_address: str):
        self.web3 = web3
        self.contract_address = contract_address
        self.contract = self.web3.eth.contract(
            address=self.contract_address, abi=self.get_abi()
        )

    def get_abi(self) -> list:
        # Load the PI KYC ABI from a file or database
        with open("pi_kyc.abi", "r") as f:
            return json.load(f)

    def perform_kyc(self, user: str, kyc_data: dict) -> bool:
        # Perform a KYC check
        tx_hash = self.contract.functions.performKYC(user, kyc_data).transact(
            {"from": self.web3.eth.accounts[0]}
        )
        return self.web3.eth.waitForTransactionReceipt(tx_hash).status == 1

    def get_kyc_status(self, user: str) -> bool:
        # Get the KYC status of a user
        return self.contract.functions.getKYCStatus(user).call()

    def set_kyc_parameters(self, parameters: dict) -> bool:
        # Set the KYC parameters
        tx_hash = self.contract.functions.setKYCParameters(parameters).transact(
            {"from": self.web3.eth.accounts[0]}
        )
        return self.web3.eth.waitForTransactionReceipt(tx_hash).status == 1
