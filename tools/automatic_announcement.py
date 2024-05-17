import datetime
import time

import pytz
import schedule
from twilio.rest import Client

# Twilio account credentials
account_sid = "your_account_sid"
auth_token = "your_auth_token"
client = Client(account_sid, auth_token)

# Set the event date and time
event_date = datetime.datetime(2024, 6, 1, 0, 0, 0, tzinfo=pytz.UTC)

# Set the message to be sent
message = "Congratulations from KOSASIH (Pi Nexus Owner) on the launch of Pi Network Open Mainet on 1 June 2024!"


# Function to send automatic announcements
def send_announcement():
    # Get the current date and time
    current_date = datetime.datetime.now(pytz.UTC)

    # Check if the event date has arrived
    if current_date >= event_date:
        # Get a list of global users (assuming you have a database or API to fetch users)
        users = get_global_users()

        # Send a message to each user
        for user in users:
            # Use Twilio to send an SMS or WhatsApp message
            message = client.messages.create(
                body=message, from_="your_twilio_number", to=user["phone_number"]
            )
            print(f"Message sent to {user['name']} ({user['phone_number']})")


# Function to get global users (replace with your own implementation)
def get_global_users():
    # Return a list of users with their names and phone numbers
    return [
        {"name": "John Doe", "phone_number": "+1234567890"},
        {"name": "Jane Doe", "phone_number": "+9876543210"},
        # Add more users here
    ]


# Schedule the announcement to be sent at the event date and time


def job():
    send_announcement()


schedule.every().day.at("00:00").do(job)  # Run the job daily at 12:00 AM

while True:
    schedule.run_pending()
    time.sleep(1)
