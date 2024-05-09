import unittest
import agents.agent_a

class TestAgentA(unittest.TestCase):
    def test_method_a(self):
        agent = agents.agent_a.AgentA()
        result = agent.method_a()
        self.assertEqual(result, 'expected result')

    def test_method_b(self):
        agent = agents.agent_a.AgentA()
        result = agent.method_b()
        self.assertIsNone(result)

if __name__ == '__main__':
    unittest.main()
