# nexus_autonomous_banking_network/api/routes.py
from fastapi import APIRouter, HTTPException
from nexus_autonomous_banking_network.core.config import settings
from nexus_autonomous_banking_network.models import User

router = APIRouter()

@router.post("/users/")
async def create_user(user: User):
    """Create a new user"""
    try:
        # Check if user already exists
        existing_user = await User.get(user.username)
        if existing_user:
            raise HTTPException(status_code=400, detail="User already exists")
        
        # Create new user
        new_user = await User.create(**user.dict())
        return new_user
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error creating user: {e}")

@router.get("/users/{username}")
async def get_user(username: str):
    """Get a user by username"""
    try:
        user = await User.get(username)
        if not user:
            raise HTTPException(status_code=404, detail="User not found")
        return user
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error getting user: {e}")
