import sqlite3
from datetime import datetime

class TransactionHistoryManager:
    def __init__(self, db_file="transaction_history.db"):
        self.conn = sqlite3.connect(db_file)
        self.cursor = self.conn.cursor()
        self.create_table()

    def create_table(self):
        self.cursor.execute("""
            CREATE TABLE IF NOT EXISTS transaction_history (
                id INTEGER PRIMARY KEY,
                transaction_id TEXT,
                user_id TEXT,
                fiat_currency TEXT,
                pi_amount REAL,
                fiat_amount REAL,
                transaction_type TEXT,
                transaction_status TEXT,
                created_at TEXT DEFAULT CURRENT_TIMESTAMP
            )
        """)
        self.conn.commit()

    def store_transaction(self, transaction_id, user_id, fiat_currency, pi_amount, fiat_amount, transaction_type, transaction_status):
        self.cursor.execute("""
            INSERT INTO transaction_history (
                transaction_id,
                user_id,
                fiat_currency,
                pi_amount,
                fiat_amount,
                transaction_type,
                transaction_status
            ) VALUES (?, ?, ?, ?, ?, ?, ?)
        """, (
            transaction_id,
            user_id,
            fiat_currency,
            pi_amount,
            fiat_amount,
            transaction_type,
            transaction_status
        ))
        self.conn.commit()

    def get_transaction_history(self, user_id=None, fiat_currency=None, transaction_type=None):
        query = "SELECT * FROM transaction_history"
        params = []
        if user_id:
            query += " WHERE user_id = ?"
            params.append(user_id)
        if fiat_currency:
            query += " AND fiat_currency = ?"
            params.append(fiat_currency)
        if transaction_type:
            query += " AND transaction_type = ?"
            params.append(transaction_type)
        self.cursor.execute(query, params)
        return self.cursor.fetchall()

    def get_transaction(self, transaction_id):
        self.cursor.execute("SELECT * FROM transaction_history WHERE transaction_id = ?", (transaction_id,))
        return self.cursor.fetchone()

    def update_transaction_status(self, transaction_id, transaction_status):
        self.cursor.execute("UPDATE transaction_history SET transaction_status = ? WHERE transaction_id = ?", (transaction_status, transaction_id))
        self.conn.commit()

    def close(self):
        self.conn.close()

def main():
    thm = TransactionHistoryManager()
    thm.store_transaction("TX123", "USER123", "USD", 100, 1000, "DEPOSIT", "PENDING")
    thm.store_transaction("TX124", "USER123", "EUR", 50, 500, "WITHDRAWAL", "COMPLETED")
    print(thm.get_transaction_history(user_id="USER123"))
    print(thm.get_transaction("TX123"))
    thm.update_transaction_status("TX123", "COMPLETED")
    print(thm.get_transaction("TX123"))
    thm.close()

if __name__ == "__main__":
    main()
