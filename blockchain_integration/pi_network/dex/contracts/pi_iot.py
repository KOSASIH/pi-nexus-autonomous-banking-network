# pi_iot.py

import web3
from web3.contract import Contract


class PIIoT:
    def __init__(self, web3: web3.Web3, contract_address: str):
        self.web3 = web3
        self.contract_address = contract_address
        self.contract = self.web3.eth.contract(
            address=self.contract_address, abi=self.get_abi()
        )

    def get_abi(self) -> list:
        # Load the PI IoT ABI from a file or database
        with open("pi_iot.abi", "r") as f:
            return json.load(f)

    def create_iot_device(self, device_data: dict) -> bool:
        # Create a new IoT device
        tx_hash = self.contract.functions.createIoTDevice(device_data).transact(
            {"from": self.web3.eth.accounts[0]}
        )
        return self.web3.eth.waitForTransactionReceipt(tx_hash).status == 1

    def send_iot_data(self, device_id: int, data: dict) -> bool:
        # Send data from an IoT device
        tx_hash = self.contract.functions.sendIoTData(device_id, data).transact(
            {"from": self.web3.eth.accounts[0]}
        )
        return self.web3.eth.waitForTransactionReceipt(tx_hash).status == 1

    def get_iot_device_details(self, device_id: int) -> dict:
        # Get the details of an IoT device
        return self.contract.functions.getIoTDeviceDetails(device_id).call()

    def set_iot_parameters(self, parameters: dict) -> bool:
        # Set the parameters for the IoT contract
        tx_hash = self.contract.functions.setIoTParameters(parameters).transact(
            {"from": self.web3.eth.accounts[0]}
        )
        return self.web3.eth.waitForTransactionReceipt(tx_hash).status == 1
