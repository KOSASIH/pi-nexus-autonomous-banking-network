import sqlite3

class DatabaseModel:
    def __init__(self, db_name: str):
        self.conn = sqlite3.connect(db_name)
        self.cursor = self.conn.cursor()

    def create_table(self, table_name: str, columns: List[str]) -> None:
        # Create a new table
        self.cursor.execute(f"CREATE TABLE IF NOT EXISTS {table_name} ({', '.join(columns)})")

    def insert_data(self, table_name: str, data: List) -> None:
        # Insert data into a table
        self.cursor.execute(f"INSERT INTO {table_name} VALUES ({', '.join(['?'] * len(data))})", data)
        self.conn.commit()

    def retrieve_data(self, table_name: str, conditions: List[str]) -> List:
        # Retrieve data from a table
        self.cursor.execute(f"SELECT * FROM {table_name} WHERE {' AND '.join(conditions)}")
        return self.cursor.fetchall()

    def close_connection(self) -> None:
        # Close the database connection
        self.conn.close()
