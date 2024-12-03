class Governance:
    def __init__(self):
        self.proposals = {}  # proposal_id -> Proposal
        self.proposal_count = 0

    def create_proposal(self, title, description, proposer):
        self.proposal_count += 1
        proposal_id = self.proposal_count
        proposal = Proposal(proposal_id, title, description, proposer)
        self.proposals[proposal_id] = proposal
        print(f"Proposal created: {title} (ID: {proposal_id}) by {proposer}.")
        return proposal_id

    def get_proposal(self, proposal_id):
        return self.proposals.get(proposal_id)

    def get_all_proposals(self):
        return self.proposals.values()

class Proposal:
    def __init__(self, proposal_id, title, description, proposer):
        self.proposal_id = proposal_id
        self.title = title
        self.description = description
        self.proposer = proposer
        self.votes = {}  # voter -> vote (True for yes, False for no)
        self.status = "Pending"

    def vote(self, voter, support):
        if self.status != "Pending":
            print("Voting is closed for this proposal.")
            return
        self.votes[voter] = support
        print(f"{voter} voted {'yes' if support else 'no'} on proposal {self.proposal_id}.")

    def tally_votes(self):
        if self.status != "Pending":
            print("Voting is closed for this proposal.")
            return
        yes_votes = sum(1 for vote in self.votes.values() if vote)
        no_votes = len(self.votes) - yes_votes
        self.status = "Passed" if yes_votes > no_votes else "Rejected"
        print(f"Proposal {self.proposal_id} has been {self.status}. Yes: {yes_votes}, No: {no_votes}.")
