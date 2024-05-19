# config/constants.py

# API Endpoints
API_VERSION = "v1"
API_BASE_URL = f"/api/{API_VERSION}"
MARKETPLACE_ENDPOINT = f"{API_BASE_URL}/marketplace"
ORDERS_ENDPOINT = f"{API_BASE_URL}/orders"
PRODUCTS_ENDPOINT = f"{API_BASE_URL}/products"
USERS_ENDPOINT = f"{API_BASE_URL}/users"

# Database
DATABASE_URL = "postgresql://username:password@localhost/silkroad"
DATABASE_NAME = "silkroad"
DATABASE_USER = "username"
DATABASE_PASSWORD = "password"
DATABASE_HOST = "localhost"
DATABASE_PORT = 5432

# Blockchain
BLOCKCHAIN_RPC_URL = "http://localhost:8545"
BLOCKCHAIN_PRIVATE_KEY = "0x..."
