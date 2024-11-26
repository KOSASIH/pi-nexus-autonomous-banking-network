import unittest
from src.core.consensus import Consensus

class TestConsensus(unittest.TestCase):
    def setUp(self):
        self.consensus = Consensus()

    def test_election(self):
        winner = self.consensus.elect_leader()
        self.assertIn(winner, self.consensus.nodes)

    def test_consensus_algorithm(self):
        result = self.consensus.run_algorithm()
        self.assertTrue(result)

if __name__ == '__main__':
    unittest.main()
