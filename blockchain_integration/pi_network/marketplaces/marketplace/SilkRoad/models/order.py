# models/order.py

from datetime import datetime
from typing import Optional

from pydantic import BaseModel


class Order(BaseModel):
    id: int
    user_id: int
    product_id: int
    quantity: int
    price: float
    status: str
    created_at: datetime
    updated_at: datetime

    class Config:
        orm_mode = True
