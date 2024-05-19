# models/user.py

from datetime import datetime
from typing import Optional

from pydantic import BaseModel

class User(BaseModel):
    id: int
    username: str
    email: str
    created_at: datetime
    updated_at: datetime

    class Config:
        orm_mode = True
