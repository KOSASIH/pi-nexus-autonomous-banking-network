import unittest
from security.config.security_config import SECURITY_CONFIG
from security.audits.audit_logger import audit_logger

class TestSecurity(unittest.TestCase):
    def test_encrypt_data(self):
        data = 'Hello, World!'
        encrypted_data = encrypt_data(data)
        decrypted_data = decrypt_data(encrypted_data)
        self.assertEqual(data, decrypted_data)

    def test_authenticate_request(self):
        request = {'headers': {'Authorization': 'Bearer SECRET_KEY'}}
        authenticated = authenticate_request(request)
        self.assertTrue(authenticated)

    def test_rate_limit(self):
        request = {'headers': {'X-Forwarded-For': '192.168.1.1'}}
        rate_limited = rate_limit(request)
        self.assertFalse(rate_limited)

    def test_audit_logger(self):
        event_type = 'TEST_EVENT'
        event_data = {'test_data': 'Hello, World!'}
        audit_logger.log_event(event_type, event_data)
        with open('audit_report.json', 'r') as f:
            audit_report = json.load(f)
            self.assertEqual(audit_report['event_type'], event_type)
            self.assertEqual(audit_report['event_data'], event_data)

if __name__ == '__main__':
    unittest.main()
