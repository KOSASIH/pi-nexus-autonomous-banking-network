import sqlite3


class MicrotransactionHandler:

    def __init__(self, db_file):
        self.db_file = db_file
        self.conn = sqlite3.connect(db_file)
        self.cursor = self.conn.cursor()

    def create_microtransaction(self, transaction_id, amount):
        # Create a new microtransaction in the database
        self.cursor.execute(
            "INSERT INTO microtransactions (transaction_id, amount) VALUES (?, ?)",
            (transaction_id, amount),
        )
        self.conn.commit()

    def get_microtransaction(self, transaction_id):
        # Retrieve a microtransaction from the database
        self.cursor.execute(
            "SELECT * FROM microtransactions WHERE transaction_id = ?",
            (transaction_id,),
        )
        return self.cursor.fetchone()
