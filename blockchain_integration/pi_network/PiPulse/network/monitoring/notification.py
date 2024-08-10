import smtplib
from email.mime.text import MIMEText

class NotificationSender:
    def __init__(self, email_config):
        self.email_config = email_config

    def send_notification(self, message):
        msg = MIMEText(message)
        msg['Subject'] = 'System Monitoring Alert'
        msg['From'] = self.email_config['from']
        msg['To'] = self.email_config['to']

        server = smtplib.SMTP(self.email_config['smtp_server'])
        server.starttls()
        server.login(self.email_config['username'], self.email_config['password'])
        server.sendmail(self.email_config['from'], self.email_config['to'], msg.as_string())
        server.quit()

if __name__ == '__main__':
    email_config = {
        'from': 'monitor@example.com',
        'to': 'admin@example.com',
        'smtp_server': 'smtp.example.com',
        'username': 'monitor',
        'password': 'password'
    }
    notification_sender = NotificationSender(email_config)
    notification_sender.send_notification('Test notification')
