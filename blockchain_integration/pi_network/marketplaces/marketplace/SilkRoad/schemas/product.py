# schemas/product.py

from pydantic import BaseModel

class ProductBase(BaseModel):
    name: str
    description: str
    price: float
    marketplace_id: int

class ProductCreate(ProductBase):
    pass

class ProductUpdate(ProductBase):
    pass

class Product(ProductBase):
    id: int

    class Config:
        orm_mode = True
