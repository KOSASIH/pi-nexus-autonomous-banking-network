# services/order_service.py

from typing import List

from schemas import OrderCreate, OrderUpdate
from sqlalchemy.orm import Session

from models import Order


def create_order(db: Session, order: OrderCreate) -> Order:
    db_order = Order(
        user_id=order.user_id,
        product_id=order.product_id,
        quantity=order.quantity,
        total_price=order.total_price,
    )
    db.add(db_order)
    db.commit()
    db.refresh(db_order)
    return db_order


def update_order(db: Session, order_id: int, order: OrderUpdate) -> Order:
    db_order = db.query(Order).filter(Order.id == order_id).first()
    if not db_order:
        return None
    db_order.user_id = order.user_id
    db_order.product_id = order.product_id
    db_order.quantity = order.quantity
    db_order.total_price = order.total_price
    db.commit()
    return db_order


def delete_order(db: Session, order_id: int) -> None:
    db_order = db.query(Order).filter(Order.id == order_id).first()
    if not db_order:
        return None
    db.delete(db_order)
    db.commit()


def get_order_by_id(db: Session, order_id: int) -> Order:
    return db.query(Order).filter(Order.id == order_id).first()


def get_orders(db: Session) -> List[Order]:
    return db.query(Order).all()
