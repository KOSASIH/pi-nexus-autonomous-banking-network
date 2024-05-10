import sqlite3


class Database:
    def __init__(self):
        self.connection = sqlite3.connect("banking.db")
        self.cursor = self.connection.cursor()

    def create_table(self):
        self.cursor.execute(
            """CREATE TABLE IF NOT EXISTS accounts (id INTEGER PRIMARY KEY, name TEXT, balance REAL)"""
        )
        self.connection.commit()

    def insert_account(self, name, balance):
        self.cursor.execute(
            "INSERT INTO accounts (name, balance) VALUES (?, ?)", (name, balance)
        )
        self.connection.commit()

    def update_account(self, id, name, balance):
        self.cursor.execute(
            "UPDATE accounts SET name=?, balance=? WHERE id=?", (name, balance, id)
        )
        self.connection.commit()

    def delete_account(self, id):
        self.cursor.execute("DELETE FROM accounts WHERE id=?", (id,))
        self.connection.commit()

    def get_account(self, id):
        self.cursor.execute("SELECT * FROM accounts WHERE id=?", (id,))
        return self.cursor.fetchone()

    def get_all_accounts(self):
        self.cursor.execute("SELECT * FROM accounts")
        return self.cursor.fetchall()
