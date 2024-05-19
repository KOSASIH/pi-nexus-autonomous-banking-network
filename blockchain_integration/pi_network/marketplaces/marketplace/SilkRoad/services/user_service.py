# services/user_service.py

from typing import List

from schemas import UserCreate, UserUpdate
from sqlalchemy.orm import Session

from models import User


def create_user(db: Session, user: UserCreate) -> User:
    db_user = User(username=user.username, email=user.email, password=user.password)
    db.add(db_user)
    db.commit()
    db.refresh(db_user)
    return db_user


def update_user(db: Session, user_id: int, user: UserUpdate) -> User:
    db_user = db.query(User).filter(User.id == user_id).first()
    if not db_user:
        return None
    db_user.username = user.username
    db_user.email = user.email
    db_user.password = user.password
    db.commit()
    return db_user


def delete_user(db: Session, user_id: int) -> None:
    db_user = db.query(User).filter(User.id == user_id).first()
    if not db_user:
        return None
    db.delete(db_user)
    db.commit()


def get_user_by_id(db: Session, user_id: int) -> User:
    return db.query(User).filter(User.id == user_id).first()


def get_users(db: Session) -> List[User]:
    return db.query(User).all()
