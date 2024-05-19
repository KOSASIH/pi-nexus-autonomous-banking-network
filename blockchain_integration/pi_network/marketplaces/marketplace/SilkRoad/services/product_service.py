# services/product_service.py

from typing import List

from schemas import ProductCreate, ProductUpdate
from sqlalchemy.orm import Session

from models import Product


def create_product(db: Session, product: ProductCreate) -> Product:
    db_product = Product(
        name=product.name,
        description=product.description,
        price=product.price,
        marketplace_id=product.marketplace_id,
    )
    db.add(db_product)
    db.commit()
    db.refresh(db_product)
    return db_product


def update_product(db: Session, product_id: int, product: ProductUpdate) -> Product:
    db_product = db.query(Product).filter(Product.id == product_id).first()
    if not db_product:
        return None
    db_product.name = product.name
    db_product.description = product.description
    db_product.price = product.price
    db_product.marketplace_id = product.marketplace_id
    db.commit()
    return db_product


def delete_product(db: Session, product_id: int) -> None:
    db_product = db.query(Product).filter(Product.id == product_id).first()
    if not db_product:
        return None
    db.delete(db_product)
    db.commit()


def get_product_by_id(db: Session, product_id: int) -> Product:
    return db.query(Product).filter(Product.id == product_id).first()


def get_products(db: Session) -> List[Product]:
    return db.query(Product).all()
