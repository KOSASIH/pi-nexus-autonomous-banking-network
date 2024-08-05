import unittest
from security.audits.audit_report import AuditReport

class TestAuditReport(unittest.TestCase):
    def test_audit_report(self):
        audit_id = 'AUDIT_ID'
        timestamp = datetime.datetime.now()
        event_type = 'TEST_EVENT'
        event_data = {'test_data': 'Hello, World!'}
        audit_report = AuditReport(audit_id, timestamp, event_type, event_data)
        self.assertEqual(audit_report.audit_id, audit_id)
        self.assertEqual(audit_report.timestamp, timestamp)
                self.assertEqual(audit_report.event_type, event_type)
        self.assertEqual(audit_report.event_data, event_data)

    def test_audit_report_to_json(self):
        audit_id = 'AUDIT_ID'
        timestamp = datetime.datetime.now()
        event_type = 'TEST_EVENT'
        event_data = {'test_data': 'Hello, World!'}
        audit_report = AuditReport(audit_id, timestamp, event_type, event_data)
        json_report = audit_report.to_json()
        self.assertEqual(json_report['audit_id'], audit_id)
        self.assertEqual(json_report['timestamp'], timestamp.isoformat())
        self.assertEqual(json_report['event_type'], event_type)
        self.assertEqual(json_report['event_data'], event_data)

if __name__ == '__main__':
    unittest.main()
