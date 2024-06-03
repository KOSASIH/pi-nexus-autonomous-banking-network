import smtplib
from email.mime.text import MIMEText

class NotificationSystem:
    def __init__(self, smtp_server, smtp_port, sender_email, sender_password):
        self.smtp_server = smtp_server
        self.smtp_port = smtp_port
        self.sender_email = sender_email
        self.sender_password = sender_password

    def send_notification(self, recipient_email, subject, message):
        # Create a text message
        msg = MIMEText(message)
        msg["Subject"] = subject
        msg["From"] = self.sender_email
        msg["To"] = recipient_email

        # Send the message using SMTP
        server = smtplib.SMTP_SSL(self.smtp_server, self.smtp_port)
        server.login(self.sender_email, self.sender_password)
        server.sendmail(self.sender_email, recipient_email, msg.as_string())
        server.quit()

if __name__ == '__main__':
    smtp_server = "smtp.gmail.com"
    smtp_port = 465
    sender_email = "your_email@gmail.com"
    sender_password = "your_password"

    notification_system = NotificationSystem(smtp_server, smtp_port, sender_email, sender_password)
    recipient_email = "recipient_email@gmail.com"
    subject = "Wallet Notification"
    message = "This is a notification from your wallet app."

    notification_system.send_notification(recipient_email, subject, message)
