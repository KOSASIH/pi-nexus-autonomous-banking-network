# pi_nexus-autonomous-banking-network/utils/database.py
import sqlite3


def connect_to_db():
    """Connect to the database"""
    conn = sqlite3.connect("database.db")
    return conn


def execute_query(conn, query):
    """Execute a query on the database"""
    cursor = conn.cursor()
    cursor.execute(query)
    return cursor.fetchall()
