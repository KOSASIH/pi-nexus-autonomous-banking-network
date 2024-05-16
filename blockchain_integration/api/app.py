# api/app.py
from fastapi import FastAPI, HTTPException

from blockchain_integration.blockchain import Blockchain

app = FastAPI()

blockchain = Blockchain("https://mainnet.infura.io/v3/YOUR_PROJECT_ID")


@app.get("/balance/{address}")
async def get_balance(address: str):
    balance = blockchain.get_balance(address)
    if balance is None:
        raise HTTPException(status_code=500, detail="Error retrieving balance")
    return {"balance": balance}
