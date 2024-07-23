from fastapi import FastAPI, HTTPException
from sidra_chain_sdk import SidraChain
from sidra_chain_db.database import Database
from sidra_chain_ml.machine_learning import MachineLearning

app = FastAPI()

sidra_chain = SidraChain()
database = Database()
machine_learning = MachineLearning()

@app.get("/healthcheck")
async def healthcheck():
    return {"status": "ok"}

@app.post("/transactions")
async def create_transaction(transaction: dict):
    try:
        sidra_chain.create_transaction(transaction)
        database.save_transaction(transaction)
        machine_learning.train_model(transaction)
        return {"status": "success"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
