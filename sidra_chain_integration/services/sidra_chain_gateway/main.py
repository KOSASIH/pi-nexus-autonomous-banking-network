# sidra_chain_gateway/main.py
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

app = FastAPI()

class RequestData(BaseModel):
    user_id: str
    transaction_data: str

@app.post("/transactions")
async def create_transaction(request_data: RequestData):
    # Route to sidra_smart_contract_orchestrator
    async with aiohttp.ClientSession() as session:
        async with session.post("http://sidra_smart_contract_orchestrator:8080/transactions", json=request_data.dict()) as response:
            return response.json()

@app.get("/users/{user_id}")
async def get_user_data(user_id: str):
    # Route to sidra_data_analytics_engine
    async with aiohttp.ClientSession() as session:
        async with session.get(f"http://sidra_data_analytics_engine:3000/users/{user_id}") as response:
            return response.json()
