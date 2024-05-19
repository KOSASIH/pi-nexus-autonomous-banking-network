# schemas/marketplace.py

from pydantic import BaseModel


class MarketplaceBase(BaseModel):
    name: str
    description: str
    location: str


class MarketplaceCreate(MarketplaceBase):
    pass


class MarketplaceUpdate(MarketplaceBase):
    pass


class Marketplace(MarketplaceBase):
    id: int

    class Config:
        orm_mode = True
