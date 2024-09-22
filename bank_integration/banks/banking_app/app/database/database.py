import os
from sqlalchemy import create_engine, event
from sqlalchemy.orm import sessionmaker, scoped_session
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.pool import QueuePool
from sqlalchemy.exc import OperationalError, SQLAlchemyError
from logging import getLogger

logger = getLogger(__name__)

class Database:
    def __init__(self):
        self.engine = self.create_engine()
        self.Session = self.create_session()

    def create_engine(self):
        # Create a SQLAlchemy engine
        database_url = os.environ.get("DATABASE_URL")
        engine = create_engine(database_url, pool_size=20, max_overflow=10, poolclass=QueuePool)
        event.listen(engine, "handle_error", self.handle_error)
        return engine

    def create_session(self):
        # Create a SQLAlchemy session
        return scoped_session(sessionmaker(bind=self.engine))

    def create_all_tables(self):
        # Create all tables in the database
        BaseModel.metadata.create_all(self.engine)

    def drop_all_tables(self):
        # Drop all tables in the database
        BaseModel.metadata.drop_all(self.engine)

    def handle_error(self, context):
        # Handle database errors
        logger.error("Database error: %s", context.original_exception)

    def execute_sql(self, sql):
        # Execute a SQL query
        with self.engine.connect() as connection:
            result = connection.execute(sql)
            return result.fetchall()

    def execute_sql_with_params(self, sql, params):
        # Execute a SQL query with parameters
        with self.engine.connect() as connection:
            result = connection.execute(sql, params)
            return result.fetchall()

    def get_table_names(self):
        # Get a list of table names in the database
        with self.engine.connect() as connection:
            result = connection.execute("SHOW TABLES")
            return [row[0] for row in result.fetchall()]

    def get_column_names(self, table_name):
        # Get a list of column names for a table
        with self.engine.connect() as connection:
            result = connection.execute(f"DESCRIBE {table_name}")
            return [row[0] for row in result.fetchall()]

    def truncate_table(self, table_name):
        # Truncate a table
        with self.engine.connect() as connection:
            connection.execute(f"TRUNCATE TABLE {table_name}")

    def vacuum(self):
        # Vacuum the database
        with self.engine.connect() as connection:
            connection.execute("VACUUM")

    def optimize(self):
        # Optimize the database
        with self.engine.connect() as connection:
            connection.execute("OPTIMIZE TABLES")
