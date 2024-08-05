import logging

# Define audit logger
class AuditLogger:
    def __init__(self):
        self.logger = logging.getLogger('audit_logger')
        self.logger.setLevel(logging.INFO)

    def log_event(self, event_type, event_data):
        audit_report = generate_audit_report(event_type, event_data)
        self.logger.info(audit_report)
        save_audit_report(audit_report)

# Define audit logger instance
audit_logger = AuditLogger()
