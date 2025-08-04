import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText

from config import get_config


def send_email_alert(exception):
    # Get email configuration from config file
    email_to = get_config("exceptions.email_to")
    email_subject = get_config("exceptions.email_subject")
    email_body = get_config("exceptions.email_body")

    # Set up email message
    msg = MIMEMultipart()
    msg["From"] = "your-email@example.com"
    msg["To"] = email_to
    msg["Subject"] = email_subject
    msg.attach(MIMEText(f"An exception occurred:\n\n{exception}\n\n{email_body}"))

    # Send email using SMTP server
    server = smtplib.SMTP("smtp.example.com", 587)
    server.starttls()
    server.login("your-email@example.com", "your-password")
    server.sendmail("your-email@example.com", email_to, msg.as_string())
    server.quit()
