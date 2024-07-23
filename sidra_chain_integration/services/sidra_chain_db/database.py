import sqlite3

class Database:
    def __init__(self):
        self.conn = sqlite3.connect("sidra_chain.db")
        self.cursor = self.conn.cursor()

    def save_transaction(self, transaction: dict):
        self.cursor.execute("INSERT INTO transactions (id, amount, sender, receiver) VALUES (?, ?, ?, ?)",
                            (transaction["id"], transaction["amount"], transaction["sender"], transaction["receiver"]))
        self.conn.commit()
