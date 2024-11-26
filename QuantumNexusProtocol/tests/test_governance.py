import unittest
from src.governance.voting_system import VotingSystem

class TestGovernance(unittest.TestCase):
    def setUp(self):
        self.voting_system = VotingSystem()

    def test_propose_change(self):
        result = self.voting_system.propose_change("Change Proposal")
        self.assertTrue(result)

    def test_vote_on_proposal(self):
        self.voting_system.propose_change("Change Proposal")
        result = self.voting_system.vote("Change Proposal", "address1", True)
        self.assertTrue(result)

if __name__ == '__main__':
    unittest.main()
