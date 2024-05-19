# pi_ticket.py - Advanced PI Coin Ticket Payment System

import os
import json
import requests
from datetime import datetime
from hashlib import sha256
import logging

# Configuration
PI_COIN_API_KEY = os.environ['PI_COIN_API_KEY']
PI_COIN_API_SECRET = os.environ['PI_COIN_API_SECRET']
TICKET_PRICE = 10  # in PI Coins
DB_FILE = 'tickets.db'  # SQLite database file

# Logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class TicketSystem:
    def __init__(self):
        self.db = self.init_db()

    def init_db(self):
        """Initialize SQLite database"""
        import sqlite3
        conn = sqlite3.connect(DB_FILE)
        c = conn.cursor()
        c.execute('''CREATE TABLE IF NOT EXISTS tickets
                     (id TEXT PRIMARY KEY, user_id INTEGER, type TEXT, price REAL, status TEXT)''')
        conn.commit()
        conn.close()
        return conn

    def generate_ticket_id(self):
        """Generate a unique ticket ID"""
        return sha256(str(datetime.now()).encode()).hexdigest()[:12]

    def create_ticket(self, user_id, ticket_type):
        """Create a new ticket"""
        ticket_id = self.generate_ticket_id()
        ticket_data = {
            'id': ticket_id,
            'user_id': user_id,
            'type': ticket_type,
            'price': TICKET_PRICE,
            'status': 'pending'
        }
        self.db.execute("INSERT INTO tickets VALUES (?, ?, ?, ?, ?)",
                        (ticket_id, user_id, ticket_type, TICKET_PRICE, 'pending'))
        self.db.commit()
        return ticket_data

    def get_pi_coin_balance(self, user_id):
        """Get the user's PI Coin balance"""
        headers = {
            'API-KEY': PI_COIN_API_KEY,
            'API-SECRET': PI_COIN_API_SECRET
        }
        response = requests.get(f'https://api.pi-coin.com/v1/users/{user_id}/balance', headers=headers)
        response.raise_for_status()  # Raise an exception for bad status codes
        return response.json()['balance']

    def pay_ticket(self, ticket_id, user_id):
        """Pay for a ticket using PI Coins"""
        ticket_data = self.get_ticket(ticket_id)
        if ticket_data['status'] != 'pending':
            logger.error(f'Ticket {ticket_id} is not pending')
            return False
        user_balance = self.get_pi_coin_balance(user_id)
        if user_balance < ticket_data['price']:
            logger.error(f'Insufficient balance for user {user_id}')
            return False
        # Perform payment using PI Coin API
        headers = {
            'API-KEY': PI_COIN_API_KEY,
            'API-SECRET': PI_COIN_API_SECRET
        }
        response = requests.post(f'https://api.pi-coin.com/v1/payments', headers=headers, json={
            'amount': ticket_data['price'],
            'user_id': user_id,
            'ticket_id': ticket_id
        })
        response.raise_for_status()  # Raise an exception for bad status codes
        if response.status_code == 200:
            self.update_ticket_status(ticket_id, 'paid')
            logger.info(f'Ticket {ticket_id} paid successfully')
            return True
        logger.error(f'Payment failed for ticket {ticket_id}')
        return False

    def get_ticket(self, ticket_id):
        """Get a ticket by ID"""
        self.db.execute("SELECT * FROM tickets WHERE id=?", (ticket_id,))
        row = self.db.fetchone()
        if row:
            return {
                'id': row[0],
                'user_id': row[1],
                'type': row[2],
                'price': row[3],
                'status': row[4]
            }
        return None

    def update_ticket_status(self, ticket_id, status):
        """Update a ticket's status"""
        self.db.execute("UPDATE tickets SET status=? WHERE id=?", (status, ticket_id))
        self.db.commit()

def main():
    ticket_system = TicketSystem()
    user_id = 1
    ticket_type = 'standard'
    ticket_data = ticket_system.create_ticket(user_id, ticket_type)
    print(f'Ticket created: {ticket_data}')
    if ticket_system.pay_ticket(ticket_data['id'], user_id):
        print(f'Ticket paid: {ticket_data["id"]}')
    else:
        print(f'Payment failed: {ticket_data["id"]}')

if __name__ == '__main__':
    main()
