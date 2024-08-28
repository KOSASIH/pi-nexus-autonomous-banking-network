import hashlib

class PiNetworkCommunity:
    """
    A representation of the Pi Network community.

    Attributes:
        members (list): List of community members
        governance_contract (GovernanceContract): Governance contract for interacting with the blockchain
    """

    def __init__(self, members, governance_contract):
        self.members = members
        self.governance_contract = governance_contract

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
        # Propose a change using the governance contract
        return self.governance_contract.propose_change(proposer, change_type, change_data)

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
        # Vote on the proposal using the governance contract
        self.governance_contract.vote_on_proposal(voter, proposal_id, vote)

    def get_proposal_status(self, proposal_id):
        """
        Get the status of a proposed change.

        Args:
            proposal_id (str): ID of the proposed change

        Returns:
            status (str): Status of the proposal (e.g., pending, approved, rejected)
        """
        # Get the status of the proposal using the governance contract
        return self.governance_contract.get_proposal_status(proposal_id)
