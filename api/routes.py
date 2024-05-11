# Use a RESTful API framework like Flask or FastAPI
from fastapi import FastAPI, HTTPException

app = FastAPI()


@app.post("/transactions")
async def create_transaction(transaction: Transaction):
    """Create a new transaction"""
    # ...
