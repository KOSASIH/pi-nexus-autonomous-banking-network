import unittest
from governance import GovernanceManager  # Assuming you have a GovernanceManager class

class TestGovernanceManager(unittest.TestCase):
    def setUp(self):
        self.governance_manager = GovernanceManager()

    def test_create_proposal(self):
        proposal = self.governance_manager.create_proposal("Increase funding for project X")
        self.assertIsNotNone(proposal)
        self.assertEqual(proposal.title, "Increase funding for project X")

    def test_vote_on_proposal(self):
        proposal = self.governance_manager.create_proposal("Improve user interface")
        self.governance_manager.vote(proposal.id, "Alice", "yes")
        self.assertEqual(proposal.votes["yes"], 1)

    def test_proposal_not_found(self):
        with self.assertRaises(ValueError):
            self.governance_manager.vote("nonexistent_id", "Bob", "no")

if __name__ == "__main__":
    unittest.main()
