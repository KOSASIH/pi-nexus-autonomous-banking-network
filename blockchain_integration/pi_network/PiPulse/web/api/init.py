from fastapi import FastAPI
from auth import auth_router
from metrics import metrics_router
from nodes import nodes_router

app = FastAPI()

app.include_router(auth_router)
app.include_router(metrics_router)
app.include_router(nodes_router)
