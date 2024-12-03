class Voting:
    def __init__(self, governance):
        self.governance = governance

    def cast_vote(self, proposal_id, voter, support):
        proposal = self.governance.get_proposal(proposal_id)
        if proposal:
            proposal.vote(voter, support)
        else:
            print(f"Proposal ID {proposal_id} does not exist.")

    def tally_votes(self, proposal_id):
        proposal = self.governance.get_proposal(proposal_id)
        if proposal:
            proposal.tally_votes()
        else:
            print(f"Proposal ID {proposal_id} does not exist.")
