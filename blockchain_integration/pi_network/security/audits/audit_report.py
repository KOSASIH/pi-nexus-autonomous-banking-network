import datetime
import json

# Define audit report class
class AuditReport:
    def __init__(self, audit_id, timestamp, event_type, event_data):
        self.audit_id = audit_id
        self.timestamp = timestamp
        self.event_type = event_type
        self.event_data = event_data

    def to_json(self):
        return {
            'audit_id': self.audit_id,
            'timestamp': self.timestamp.isoformat(),
            'event_type': self.event_type,
            'event_data': self.event_data,
        }

# Define audit report generator
def generate_audit_report(event_type, event_data):
    audit_id = generate_uuid()
    timestamp = datetime.datetime.now()
    audit_report = AuditReport(audit_id, timestamp, event_type, event_data)
    return audit_report.to_json()

# Define audit report saver
def save_audit_report(audit_report):
    with open('audit_report.json', 'a') as f:
        json.dump(audit_report, f)
        f.write('\n')
