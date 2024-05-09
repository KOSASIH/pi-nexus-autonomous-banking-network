import time
import requests
import logging
import smtplib
from email.mime.text import MIMEText

def check_status_code(url):
    """Checks the HTTP status code of a URL."""
    try:
response = requests.get(url)
        if response.status_code != 200:
            logging.error(f'Error: {response.status_code}')
            send_alert(f'Error: {response.status_code}')
    except requests.exceptions.RequestException as e:
        logging.error(f'Error: {e}')
        send_alert(f'Error: {e}')

def check_response_time(url):
    """Checks the response time of a URL."""
    start_time = time.time()
    try:
        response = requests.get(url)
        end_time = time.time()
        response_time = end_time - start_time
        if response_time > 5:
            logging.warning(f'Warning: {response_time}')
            send_alert(f'Warning: {response_time}')
    except requests.exceptions.RequestException as e:
        logging.error(f'Error: {e}')
        send_alert(f'Error: {e}')

def send_alert(message):
    """Sends an alert email."""
    sender = 'sender@example.com'
    receiver = 'receiver@example.com'
    password = 'password'

    msg = MIMEText(message)
    msg['Subject'] = 'Alert'
    msg['From'] = sender
    msg['To'] = receiver

    try:
        server = smtplib.SMTP('smtp.example.com', 587)
        server.starttls()
        server.login(sender, password)
        server.sendmail(sender, receiver, msg.as_string())
        server.quit()
    except smtplib.SMTPException as e:
        logging.error(f'Error: {e}')

def main():
    """Runs the monitoring module."""
    # Set up logging
    logging.basicConfig(filename='monitoring.log', level=logging.DEBUG)

    # Monitor the status code and response time of a URL
    url = 'http://example.com'
    while True:
        check_status_code(url)
        check_response_time(url)
        time.sleep(60)

if __name__ == '__main__':
    main()
