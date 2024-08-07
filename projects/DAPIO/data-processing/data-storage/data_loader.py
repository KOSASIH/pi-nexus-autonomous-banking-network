import pandas as pd
from sqlalchemy import create_engine

class DataLoader:
    def __init__(self, db_url: str):
        self.db_url = db_url

    def load_data(self, table_name: str) -> pd.DataFrame:
        engine = create_engine(self.db_url)
        data = pd.read_sql_table(table_name, engine)
        return data

    def save_data(self, data: pd.DataFrame, table_name: str) -> None:
        engine = create_engine(self.db_url)
        data.to_sql(table_name, engine, if_exists='replace', index=False)
