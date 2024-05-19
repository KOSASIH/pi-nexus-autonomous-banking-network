# models/product.py

from datetime import datetime
from typing import Optional

from pydantic import BaseModel


class Product(BaseModel):
    id: int
    marketplace_id: int
    name: str
    description: str
    price: float
    quantity: int
    created_at: datetime
    updated_at: datetime

    class Config:
        orm_mode = True
