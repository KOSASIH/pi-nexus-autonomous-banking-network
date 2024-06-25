from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from fastapi.requests import Request

app = FastAPI()

@app.post("/register_node")
async def register_node(node_address: str):
    #...
    try:
        node = PINode(node_address, contract_address)
        await node.register_node()
        return JSONResponse(content={"message": "Node registered successfully"}, status_code=201)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/get_node_list")
async def get_node_list():
    #...
    try:
        node = PINode(node_address, contract_address)
        node_list = await node.get_node_list()
        return JSONResponse(content={"node_list": node_list}, status_code=200)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
