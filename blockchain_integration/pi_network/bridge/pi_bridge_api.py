from fastapi import FastAPI, HTTPException
from web3 import Web3
from pi_bridge_contract import PiBridgeContract

app = FastAPI()

w3 = Web3(Web3.HTTPProvider("https://mainnet.infura.io/v3/YOUR_PROJECT_ID"))
pi_bridge_contract = PiBridgeContract("0x...PiBridgeContractAddress...")

@app.post("/deposit")
async def deposit(pi_token_amount: int):
    if pi_token_amount <= 0:
        raise HTTPException(status_code=400, detail="Invalid deposit amount")
    tx_hash = pi_bridge_contract.deposit(pi_token_amount, {"from": "0x...UserAddress..."})
    return {"tx_hash": tx_hash}

@app.post("/withdraw")
async def withdraw(pi_token_amount: int):
    if pi_token_amount <= 0:
        raise HTTPException(status_code=400, detail="Invalid withdrawal amount")
    tx_hash = pi_bridge_contract.withdraw(pi_token_amount, {"from": "0x...UserAddress..."})
    return {"tx_hash": tx_hash}

@app.get("/balance")
async def get_balance():
    balance = pi_bridge_contract.userBalances("0x...UserAddress...")
    return {"balance": balance}
