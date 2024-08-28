import hashlib
from web3 import Web3

class GovernanceContract:
    """
    A smart contract for autonomous governance and voting.

    Attributes:
        web3 (Web3): Web3 provider for interacting with the Ethereum blockchain
        contract_address (str): Address of the governance contract on the Ethereum blockchain
        abi (list): ABI (Application Binary Interface) of the governance contract
    """

    def __init__(self, web3, contract_address, abi):
        self.web3 = web3
        self.contract_address = contract_address
        self.abi = abi

    def propose_change(self, proposer, change_type, change_data):
        """
        Propose a change to the network's parameters.

        Args:
            proposer (str): Address of the proposer
            change_type (str): Type of change (e.g., parameter update, new feature)
            change_data (str): Data associated with the change (e.g., new parameter value)

        Returns:
            proposal_id (str): ID of the proposed change
        """
        # Create a new proposal and store it on the blockchain
        proposal_id = hashlib.sha256(f"{proposer}{change_type}{change_data}".encode()).hexdigest()
        self.web3.eth.contract(address=self.contract_address, abi=self.abi).functions.proposeChange(proposal_id, proposer, change_type, change_data).transact()
        return proposal_id

    def vote_on_proposal(self, voter, proposal_id, vote):
        """
        Vote on a proposed change.

        Args:
            voter (str): Address of the voter
            proposal_id (str): ID of the proposed change
            vote (bool): Vote (True for yes, False for no)

        Returns:
            None
        """
        # Cast a vote on the proposal and update the vote count on the blockchain
        self.web3.eth.contract(address=self.contract_address, abi=self.abi).functions.voteOnProposal(proposal_id, voter, vote).transact()

    def get_proposal_status(self, proposal_id):
        """
        Get the status of a proposed change.

        Args:
            proposal_id (str): ID of the proposed change

        Returns:
            status (str): Status of the proposal (e.g., pending, approved, rejected)
        """
        # Retrieve the status of the proposal from the blockchain
        return self.web3.eth.contract(address=self.contract_address, abi=self.abi).functions.getProposalStatus(proposal_id).call()

    def execute_change(self, proposal_id):
        """
        Execute a approved change.

        Args:
            proposal_id (str): ID of the approved change

        Returns:
            None
        """
        # Execute the approved change and update the network's parameters
        self.web3.eth.contract(address=self.contract_address, abi=self.abi).functions.executeChange(proposal_id).transact()
