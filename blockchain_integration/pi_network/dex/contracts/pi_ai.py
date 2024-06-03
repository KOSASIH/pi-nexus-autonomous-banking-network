# pi_ai.py

import web3
from web3.contract import Contract


class PIAI:
    def __init__(self, web3: web3.Web3, contract_address: str):
        self.web3 = web3
        self.contract_address = contract_address
        self.contract = self.web3.eth.contract(
            address=self.contract_address, abi=self.get_abi()
        )

    def get_abi(self) -> list:
        # Load the PI AI ABI from a file or database
        with open("pi_ai.abi", "r") as f:
            return json.load(f)

    def create_ai_model(self, model_data: dict) -> bool:
        # Create a new AI model
        tx_hash = self.contract.functions.createAIModel(model_data).transact(
            {"from": self.web3.eth.accounts[0]}
        )
        return self.web3.eth.waitForTransactionReceipt(tx_hash).status == 1

    def train_ai_model(self, model_id: int, training_data: dict) -> bool:
        # Train an AI model
        tx_hash = self.contract.functions.trainAIModel(
            model_id, training_data
        ).transact({"from": self.web3.eth.accounts[0]})
        return self.web3.eth.waitForTransactionReceipt(tx_hash).status == 1

    def use_ai_model(self, model_id: int, input_data: dict) -> any:
        # Use an AI model
        return self.contract.functions.useAIModel(model_id, input_data).call()

    def get_ai_model_details(self, model_id: int) -> dict:
        # Get the details of an AI model
        return self.contract.functions.getAIModelDetails(model_id).call()

    def set_ai_parameters(self, parameters: dict) -> bool:
        # Set the parameters for the AI contract
        tx_hash = self.contract.functions.setAIParameters(parameters).transact(
            {"from": self.web3.eth.accounts[0]}
        )
        return self.web3.eth.waitForTransactionReceipt(tx_hash).status == 1
