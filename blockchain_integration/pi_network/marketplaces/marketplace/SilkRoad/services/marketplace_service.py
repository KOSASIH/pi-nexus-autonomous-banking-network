# services/marketplace_service.py

from typing import List
from sqlalchemy.orm import Session
from models import Marketplace
from schemas import MarketplaceCreate, MarketplaceUpdate

def create_marketplace(db: Session, marketplace: MarketplaceCreate) -> Marketplace:
    db_marketplace = Marketplace(name=marketplace.name, description=marketplace.description, location=marketplace.location)
    db.add(db_marketplace)
    db.commit()
    db.refresh(db_marketplace)
    return db_marketplace

def update_marketplace(db: Session, marketplace_id: int, marketplace: MarketplaceUpdate) -> Marketplace:
    db_marketplace = db.query(Marketplace).filter(Marketplace.id == marketplace_id).first()
    if not db_marketplace:
        return None
    db_marketplace.name = marketplace.name
    db_marketplace.description = marketplace.description
    db_marketplace.location = marketplace.location
    db.commit()
    return db_marketplace

def delete_marketplace(db: Session, marketplace_id: int) -> None:
    db_marketplace = db.query(Marketplace).filter(Marketplace.id == marketplace_id).first()
    if not db_marketplace:
        return None
    db.delete(db_marketplace)
    db.commit()

def get_marketplace_by_id(db: Session, marketplace_id: int) -> Marketplace:
    return db.query(Marketplace).filter(Marketplace.id == marketplace_id).first()

def get_marketplaces(db: Session) -> List[Marketplace]:
    return db.query(Marketplace).all()
