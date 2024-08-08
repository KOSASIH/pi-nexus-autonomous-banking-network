from sqlalchemy import create_engine
from contract_sphere_db.models import Contract, User

engine = create_engine('postgresql://user:password@localhost/contract_sphere_db')

def create_contract(title, description, user_id):
    contract = Contract(title=title, description=description, user_id=user_id)
    with engine.connect() as conn:
        conn.execute(contract.insert())
    return contract
