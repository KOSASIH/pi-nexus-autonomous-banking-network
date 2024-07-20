# did_module/did_based_autonomous_banking_api.py
import didkit
from AutonomousBankingSmartContract import AutonomousBankingSmartContract
from pi_nexus_autonomous_banking_network import AutonomousBankingNetwork


class DIDBasedAutonomousBankingAPI:
    def __init__(self, autonomous_banking_network: AutonomousBankingNetwork):
        self.autonomous_banking_network = autonomous_banking_network
        self.autonomous_banking_smart_contract = AutonomousBankingSmartContract()

    def create_account(self, did: didkit.DID):
        # Create an autonomous banking account associated with the DID
        account = self.autonomous_banking_network.create_account(did)
        return account

    def deposit(self, did: didkit.DID, amount: uint256):
        # Deposit funds into the autonomous banking account associated with the DID
        self.autonomous_banking_smart_contract.deposit(amount)
        return True

    def withdraw(self, did: didkit.DID, amount: uint256):
        # Withdraw funds from the autonomous banking account associated with the DID
        self.autonomous_banking_smart_contract.withdraw(amount)
        return True

    def transfer(self, did: didkit.DID, recipient_did: didkit.DID, amount: uint256):
        # Transfer funds to another autonomous banking account associated with the recipient DID
        self.autonomous_banking_smart_contract.transfer(recipient_did, amount)
        return True
