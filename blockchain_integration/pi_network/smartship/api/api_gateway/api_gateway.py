from fastapi import FastAPI, HTTPException, Security
from fastapi.security import OAuth2PasswordBearer, OAuth2
from fastapi.responses import JSONResponse
from fastapi.requests import Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List

app = FastAPI()

# OAuth2 authentication
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token")

# CORS configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Rate limiting using Redis
from redis import Redis
redis_client = Redis(host="redis", port=6379, db=0)

@app.middleware("http")
async def rate_limit(request: Request, call_next):
    ip_address = request.client.host
    if redis_client.get(ip_address) and int(redis_client.get(ip_address)) > 100:
        return JSONResponse(status_code=429, content={"error": "Rate limit exceeded"})
    response = await call_next(request)
    redis_client.incr(ip_address)
    redis_client.expire(ip_address, 60)  # 1 minute expiration
    return response

# API endpoints
@app.post("/login")
async def login(username: str, password: str):
    # Authenticate user and return token
    pass

@app.get("/users/")
async def read_users(token: str = Security(oauth2_scheme)):
    # Return list of users
    pass

@app.post("/transactions/")
async def create_transaction(token: str = Security(oauth2_scheme)):
    # Create a new transaction
    pass
