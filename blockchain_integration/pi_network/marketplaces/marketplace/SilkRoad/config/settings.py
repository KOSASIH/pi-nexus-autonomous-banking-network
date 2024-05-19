# config/settings.py

import os

from dotenv import load_dotenv

load_dotenv()

# General Settings
ENVIRONMENT = os.getenv("ENVIRONMENT", "development")
DEBUG = os.getenv("DEBUG", "false").lower() == "true"
SECRET_KEY = os.getenv("SECRET_KEY", "your-secret-key")

# Database Settings
DATABASE_URL = os.getenv("DATABASE_URL", constants.DATABASE_URL)
DATABASE_NAME = os.getenv("DATABASE_NAME", constants.DATABASE_NAME)
DATABASE_USER = os.getenv("DATABASE_USER", constants.DATABASE_USER)
DATABASE_PASSWORD = os.getenv("DATABASE_PASSWORD", constants.DATABASE_PASSWORD)
DATABASE_HOST = os.getenv("DATABASE_HOST", constants.DATABASE_HOST)
DATABASE_PORT = os.getenv("DATABASE_PORT", constants.DATABASE_PORT)

# Blockchain Settings
BLOCKCHAIN_RPC_URL = os.getenv("BLOCKCHAIN_RPC_URL", constants.BLOCKCHAIN_RPC_URL)
BLOCKCHAIN_PRIVATE_KEY = os.getenv(
    "BLOCKCHAIN_PRIVATE_KEY", constants.BLOCKCHAIN_PRIVATE_KEY
)
