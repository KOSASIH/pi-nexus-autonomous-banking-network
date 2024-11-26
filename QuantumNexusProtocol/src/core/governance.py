import json
from datetime import datetime
from collections import defaultdict

class Governance:
    def __init__(self):
        self.proposals = []
        self.votes = defaultdict(lambda: defaultdict(int))  # proposal_id -> {voter_id: vote_count}
        self.voting_threshold = 0.5  # 50% of votes needed to pass a proposal

    def create_proposal(self, proposal_id, description):
        """Create a new governance proposal."""
        proposal = {
            'id': proposal_id,
            'description': description,
            'created_at': datetime.utcnow().isoformat(),
            'votes_for': 0,
            'votes_against': 0,
            'status': 'pending'  # pending, passed, rejected
        }
        self.proposals.append(proposal)
        print(f"Proposal created: {proposal}")

    def vote(self, proposal_id, voter_id, vote):
        """Vote on a proposal."""
        if vote not in ['for', 'against']:
            raise ValueError("Vote must be 'for' or 'against'.")

        # Check if the proposal exists
        proposal = self.get_proposal(proposal_id)
        if not proposal:
            raise ValueError("Proposal not found.")

        # Record the vote
        if vote == 'for':
            self.votes[proposal_id][voter_id] += 1
            proposal['votes_for'] += 1
        else:
            self.votes[proposal_id][voter_id] -= 1
            proposal['votes_against'] += 1

        print(f"Vote recorded: {voter_id} voted {vote} on proposal {proposal_id}")

    def get_proposal(self, proposal_id):
        """Retrieve a proposal by its ID."""
        for proposal in self.proposals:
            if proposal['id'] == proposal_id:
                return proposal
        return None

    def tally_votes(self, proposal_id):
        """Tally votes for a proposal and determine its status."""
        proposal = self.get_proposal(proposal_id)
        if not proposal:
            raise ValueError("Proposal not found.")

        total_votes = proposal['votes_for'] + proposal['votes_against']
        if total_votes == 0:
            print("No votes have been cast for this proposal.")
            return

        if proposal['votes_for'] / total_votes >= self.voting_threshold:
            proposal['status'] = 'passed'
            print(f"Proposal {proposal_id} has passed.")
        else:
            proposal['status'] = 'rejected'
            print(f"Proposal {proposal_id} has been rejected.")

    def execute_proposal(self, proposal_id):
        """Execute a proposal if it has passed."""
        proposal = self.get_proposal(proposal_id)
        if not proposal:
            raise ValueError("Proposal not found.")

        if proposal['status'] != 'passed':
            raise ValueError("Proposal must be passed to execute.")

        # Implement the logic for executing the proposal here
        print(f"Executing proposal {proposal_id}: {proposal['description']}")

    def get_all_proposals(self):
        """Return all proposals."""
        return self.proposals

# Example usage
if __name__ == '__main__':
    governance = Governance()

    # Create proposals
    governance.create_proposal('1', 'Increase block size limit.')
    governance.create_proposal('2', 'Implement new consensus algorithm.')

    # Simulate voting
    governance.vote('1', 'Alice', 'for')
    governance.vote('1', 'Bob', 'against')
    governance.vote('1', 'Charlie', 'for')

    # Tally votes
    governance.tally_votes('1')

    # Execute proposal if passed
    governance.execute_proposal('1')

    # Retrieve all proposals
    all_proposals = governance.get_all_proposals()
    print("All Proposals:", all_proposals)
