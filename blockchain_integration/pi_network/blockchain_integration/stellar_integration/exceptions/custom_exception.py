# custom_exception.py
import logging
from logging.handlers import SMTPHandler
from email.utils import formatdate

class CustomException(Exception):
    def __init__(self, message, code, data=None):
        self.message = message
        self.code = code
        self.data = data
        self.logger = logging.getLogger(__name__)

    def __str__(self):
        return f"{self.code}: {self.message}"

    def log_exception(self):
        self.logger.error(f"Exception {self.code}: {self.message}")
        if self.data:
            self.logger.debug(f"Exception data: {self.data}")

    def send_notification(self, recipients, subject):
        mail_handler = SMTPHandler(
            mailhost=("your_smtp_server", 587),
            fromaddr="your_from_email",
            toaddrs=recipients,
            subject=subject
        )
        mail_handler.setLevel(logging.ERROR)
        self.logger.addHandler(mail_handler)
        self.log_exception()
        self.logger.removeHandler(mail_handler)
