from fastapi import APIRouter, Depends, HTTPException
from fastapi.security import OAuth2PasswordBearer, OAuth2
from jose import jwt
from pydantic import BaseModel

oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token")

class User(BaseModel):
    username: str
    email: str

class TokenData(BaseModel):
    username: str

async def authenticate_user(username: str, password: str):
    # Authenticate user against a database or other authentication system
    # For demonstration purposes, assume a successful authentication
    return User(username=username, email=f"{username}@example.com")

async def get_current_user(token: str = Depends(oauth2_scheme)):
    credentials_exception = HTTPException(
        status_code=401,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )
    try:
        payload = jwt.decode_token(token, "SECRET_KEY", algorithms=["HS256"])
        username: str = payload.get("sub")
        if username is None:
            raise credentials_exception
        token_data = TokenData(username=username)
        user = await authenticate_user(username, token_data.username)
        return user
    except jwt.JWTError:
        raise credentials_exception

auth_router = APIRouter()

@auth_router.post("/token")
async def login_for_access_token(username: str, password: str):
    user = await authenticate_user(username, password)
    access_token = jwt.create_token(user.username, "SECRET_KEY", algorithm="HS256")
    return {"access_token": access_token, "token_type": "bearer"}

@auth_router.get("/users/me")
async def read_users_me(current_user: User = Depends(get_current_user)):
    return current_user
