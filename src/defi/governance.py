from collections import defaultdict
import time

class Token:
    def __init__(self, name, symbol, total_supply):
        self.name = name
        self.symbol = symbol
        self.total_supply = total_supply
        self.balances = defaultdict(int)

    def transfer(self, from_address, to_address, amount):
        if self.balances[from_address] < amount:
            raise ValueError("Insufficient balance")
        self.balances[from_address] -= amount
        self.balances[to_address] += amount

    def mint(self, to_address, amount):
        self.balances[to_address] += amount
        self.total_supply += amount

class Proposal:
    def __init__(self, proposal_id, creator, description):
        self.proposal_id = proposal_id
        self.creator = creator
        self.description = description
        self.votes = defaultdict(int)  # user -> vote (1 for yes, -1 for no)
        self.created_at = time.time()
        self.executed = False

class Governance:
    def __init__(self, token):
        self.token = token
        self.proposals = {}
        self.proposal_id_counter = 0
        self.quorum = 0.1  # 10% of total supply must vote

    def create_proposal(self, creator, description):
        proposal = Proposal(self.proposal_id_counter, creator, description)
        self.proposals[self.proposal_id_counter] = proposal
        self.proposal_id_counter += 1
        print(f"Proposal created: {proposal.proposal_id} by {creator} - {description}")

    def vote(self, user, proposal_id, vote_value):
        if proposal_id not in self.proposals:
            raise ValueError("Proposal does not exist")
        if vote_value not in [1, -1]:
            raise ValueError("Vote must be 1 (yes) or -1 (no)")

        proposal = self.proposals[proposal_id]
        if user in proposal.votes:
            print(f"{user} has already voted on proposal {proposal_id}.")
            return

        proposal.votes[user] = vote_value
        print(f"{user} voted {'yes' if vote_value == 1 else 'no'} on proposal {proposal_id}")

    def execute_proposal(self, proposal_id):
        if proposal_id not in self.proposals:
            raise ValueError("Proposal does not exist")
        proposal = self.proposals[proposal_id]

        if proposal.executed:
            print(f"Proposal {proposal_id} has already been executed.")
            return

        total_votes = sum(proposal.votes.values())
        total_supply = self.token.total_supply
        yes_votes = sum(vote for vote in proposal.votes.values() if vote == 1)
        no_votes = sum(vote for vote in proposal.votes.values() if vote == -1)

        if len(proposal.votes) / total_supply < self.quorum:
            print(f"Proposal {proposal_id} did not meet quorum requirements.")
            return

        if yes_votes > no_votes:
            print(f"Proposal {proposal_id} passed with {yes_votes} yes votes and {no_votes} no votes.")
            # Execute the proposal logic here (e.g., upgrade contract, change parameters)
        else:
            print(f"Proposal {proposal_id} failed with {yes_votes} yes votes and {no_votes} no votes.")

        proposal.executed = True

# Example usage
if __name__ == "__main__":
    # Create a governance token
    governance_token = Token("GovernanceToken", "GOV", 1000000)

    # Mint some tokens for users
    governance_token.mint("user1", 10000)
    governance_token.mint("user2", 20000)

    # Create a governance contract
    governance = Governance(governance_token)

    # User creates a proposal
    governance.create_proposal("user1", "Increase the reward rate for yield farming.")

    # Users vote on the proposal
    governance.vote("user1", 0, 1)  # User1 votes yes
    governance.vote("user2", 0, - 1)  # User2 votes no

    # Execute the proposal
    governance.execute_proposal(0)  # Attempt to execute the proposal
