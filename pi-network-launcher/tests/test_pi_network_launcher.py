import unittest

from pi_network_launcher import PiNetworkLauncher


class TestPiNetworkLauncher(unittest.TestCase):
    def test_launcher(self):
        launcher = PiNetworkLauncher()
        output = launcher.launch()
        self.assertEqual(output, "Expected output")


if __name__ == "__main__":
    unittest.main()
