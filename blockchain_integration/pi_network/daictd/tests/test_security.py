import unittest
from security import SecurityOracle, SecurityIncidentResponse

class TestSecurityOracle(unittest.TestCase):
    def setUp(self):
        self.oracle = SecurityOracle('private_key.pem', 'public_key.pub')

    def test_generate_keys(self):
        # Test generating cryptographic keys
        self.oracle.generate_keys()
        self.assertTrue(os.path.exists('private_key.pem'))
        self.assertTrue(os.path.exists('public_key.pub'))

    def test_sign_data(self):
        # Test signing data with the private key
        data = [...]
        signature = self.oracle.sign_data(data)
        self.assertIsInstance(signature, bytes)

class TestSecurityIncidentResponse(unittest.TestCase):
    def setUp(self):
        self.incident_response = SecurityIncidentResponse(SecurityOracleAPI(SecurityOracle('private_key.pem', 'public_key.pub')))

    def test_report_incident(self):
        # Test reporting an incident and receiving a response
        incident_data = [...]
        response = self.incident_response.report_incident(incident_data)
        self.assertIsInstance(response, dict)

    def test_verify_response(self):
        # Test verifying a response signature
        response = [...]
        signature = [...]
        self.assertTrue(self.incident_response.verify_response(response, signature))

if __name__ == '__main__':
    unittest.main()
