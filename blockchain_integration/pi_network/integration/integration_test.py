import unittest
from pi_network.pi_network import PiNetwork
from stellar_integration.stellar_integration import StellarIntegration

class TestIntegration(unittest.TestCase):
    def test_pi_network_stellar_integration(self):
        pn = PiNetwork()
        si = StellarIntegration()
        pn.set_stellar_integration(si)
        account_id = "GA...123"
        balance = pn.get_account_balance(account_id)
        self.assertIsNotNone(balance)

if __name__ == '__main__':
    unittest.main()
