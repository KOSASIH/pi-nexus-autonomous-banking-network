# PI Coin Ticket Payment System

This project is a simple PI Coin ticket payment system implemented in Python. It uses the PI Coin API to handle payments and a SQLite database to store ticket information.

## Requirements

- Python 3.x
- Requests library
- SQLite3

# Configuration

1. Set up a PI Coin API account and obtain your API key and secret.
2. Set the following environment variables:
3. PI_COIN_API_KEY: Your PI Coin API key.
4. PI_COIN_API_SECRET: Your PI Coin API secret.

# Usage

Run the pi_ticket.py script to create a new ticket and process a payment.

```bash

python pi_ticket.py
```

The script will output the ticket information and whether the payment was successful or not.

# Classes and Functions

- TicketSystem: A class that handles ticket creation, payment, and status updates.
   
   - init_db: Initializes the SQLite database.
   - generate_ticket_id: Generates a unique ticket ID.
   - create_ticket: Creates a new ticket.
   - get_pi_coin_balance: Retrieves the user's PI Coin balance.
   - pay_ticket: Processes a payment for a ticket.
   - get_ticket: Retrieves a ticket by ID.
   - update_ticket_status: Updates a ticket's status.

# Error Handling

The system includes error handling for bad status codes when making API calls.

# Security

API keys are stored securely using environment variables.

# Logging

The system includes basic logging for informational and error messages.

# Testing

To test the system, you can run the pi_ticket.py script and check the output for the ticket information and payment status.

# Contributing

Pull requests are welcome. For major changes, please open an issue first to discuss what you would like to change.

# License

This project is licensed under the MIT License - see the LICENSE file for details.


                                      " Happy coding ... ☺ ...  ☕ "
