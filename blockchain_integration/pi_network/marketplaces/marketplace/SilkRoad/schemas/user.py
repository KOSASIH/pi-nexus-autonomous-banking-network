# schemas/user.py

from pydantic import BaseModel

class UserBase(BaseModel):
    username: str
    email: str
    password: str

class UserCreate(UserBase):
    pass

class UserUpdate(UserBase):
    pass

class User(UserBase):
    id: int
    orders: list[int] = []
    products: list[int] = []

    class Config:
        orm_mode = True
