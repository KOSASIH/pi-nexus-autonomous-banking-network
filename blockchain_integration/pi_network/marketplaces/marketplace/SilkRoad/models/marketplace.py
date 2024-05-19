# models/marketplace.py

from datetime import datetime
from typing import Optional

from pydantic import BaseModel

class Marketplace(BaseModel):
    id: int
    name: str
    description: str
    created_at: datetime
    updated_at: datetime

    class Config:
        orm_mode = True
