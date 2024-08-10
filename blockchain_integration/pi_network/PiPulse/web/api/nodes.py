from fastapi import APIRouter, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel

class Node(BaseModel):
    id: int
    name: str
    status: str

nodes = [
    Node(id=1, name="Node 1", status="online"),
    Node(id=2, name="Node 2", status="offline"),
    Node(id=3, name="Node 3", status="online"),
]

nodes_router = APIRouter()

@nodes_router.get("/nodes")
async def get_nodes():
    return nodes

@nodes_router.get("/nodes/{node_id}")
async def get_node(node_id: int):
    for node in nodes:
        if node.id == node_id:
            return node
    raise HTTPException(status_code=404, detail="Node not found")

@nodes_router.post("/nodes")
async def create_node(node: Node):
    nodes.append(node)
    return node

@nodes_router.put("/nodes/{node_id}")
async def update_node(node_id: int, node: Node):
    for i, existing_node in enumerate(nodes):
        if existing_node.id == node_id:
            nodes[i] = node
            return node
    raise HTTPException(status_code=404, detail="Node not found")

@nodes_router.delete("/nodes/{node_id}")
async def delete_node(node_id: int):
    for i, existing_node in enumerate(nodes):
        if existing_node.id == node_id:
            del nodes[i]
            return JSONResponse(status_code=204)
    raise HTTPException(status_code=404, detail="Node not found")
