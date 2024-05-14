from sqlalchemy import create_engine
from sqlalchemy.pool import QueuePool

engine = create_engine(
    "postgresql://user:password@host:port/dbname", poolclass=QueuePool
)


def get_transactions():
    with engine.connect() as conn:
        result = conn.execute("SELECT * FROM transactions")
        # ...
