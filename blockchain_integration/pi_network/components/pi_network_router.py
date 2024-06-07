from fastapi import FastAPI, HTTPException
from web3 import Web3
from pi_token_manager import PiTokenManager

app = FastAPI()

pi_token_manager = PiTokenManager("0x...PiTokenAddress...", "https://mainnet.infura.io/v3/YOUR_PROJECT_ID")

@app.post("/mint_tokens")
async def mint_tokens(amount: int, recipient: str):
    try:
        tx_hash = pi_token_manager.mint_tokens(amount, recipient)
        return {"tx_hash": tx_hash}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/transfer_tokens")
async def transfer_tokens(amount: int, sender: str, recipient: str):
    try:
        tx_hash = pi_token_manager.transfer_tokens(amount, sender, recipient)
        return {"tx_hash": tx_hash}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/get_token_balance")
async def get_token_balance(address: str):
    try:
        balance = pi_token_manager.get_token_balance(address)
        return {"balance": balance}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
