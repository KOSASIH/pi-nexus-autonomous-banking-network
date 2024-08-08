from sqlalchemy import create_engine
from contract_sphere_db.models import Contract

engine = create_engine('postgresql://user:password@localhost/contract_sphere_db')

def get_contracts():
    with engine.connect() as conn:
        result = conn.execute(Contract.select())
        return [dict(row) for row in result]
