from fastapi import FastAPI, HTTPException
from sidra_chain_sdk import SidraChain

app = FastAPI()

sidra_chain = SidraChain()


@app.get("/healthcheck")
async def healthcheck():
    return {"status": "ok"}


@app.post("/transactions")
async def create_transaction(transaction: dict):
    try:
        sidra_chain.create_transaction(transaction)
        return {"status": "success"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
