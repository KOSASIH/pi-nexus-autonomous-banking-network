import os

# Ethereum network configuration
ETH_NETWORK = os.environ.get("ETH_NETWORK", "rinkeby")
ETH_INFURA_PROJECT_ID = os.environ.get("ETH_INFURA_PROJECT_ID", "")
ETH_PRIVATE_KEY = os.environ.get("ETH_PRIVATE_KEY", "")

# Contract addresses
BANK_CONTRACT_ADDRESS = os.environ.get("BANK_CONTRACT_ADDRESS", "")
INTEREST_CONTRACT_ADDRESS = os.environ.get("INTEREST_CONTRACT_ADDRESS", "")
LOAN_CONTRACT_ADDRESS = os.environ.get("LOAN_CONTRACT_ADDRESS", "")
SECURITY_CONTRACT_ADDRESS = os.environ.get("SECURITY_CONTRACT_ADDRESS", "")

# Contract ABI
BANK_ABI = os.environ.get("BANK_ABI", "")
INTEREST_ABI = os.environ.get("INTEREST_ABI", "")
LOAN_ABI = os.environ.get("LOAN_ABI", "")
SECURITY_ABI = os.environ.get("SECURITY_ABI", "")

# Web3 provider
w3 = web3.Web3(
    web3.HTTPProvider(f"https://{ETH_NETWORK}.infura.io/v3/{ETH_INFURA_PROJECT_ID}")
)

# Contract instances
bank = w3.eth.contract(address=BANK_CONTRACT_ADDRESS, abi=BANK_ABI)
interest = w3.eth.contract(address=INTEREST_CONTRACT_ADDRESS, abi=INTEREST_ABI)
loan = w3.eth.contract(address=LOAN_CONTRACT_ADDRESS, abi=LOAN_ABI)
security = w3.eth.contract(address=SECURITY_CONTRACT_ADDRESS, abi=SECURITY_ABI)

# Database configuration
DB_HOST = os.getenv("DB_HOST", "localhost")
DB_PORT = int(os.getenv("DB_PORT", 5432))
DB_NAME = os.getenv("DB_NAME", "banking_network")
DB_USER = os.getenv("DB_USER", "postgres")
DB_PASSWORD = os.getenv("DB_PASSWORD", "postgres")

# API configuration
API_HOST = os.getenv("API_HOST", "0.0.0.0")
API_PORT = int(os.getenv("API_PORT", 8000))
API_DEBUG = os.getenv("API_DEBUG", "True").lower() == "true"

# Authentication configuration
AUTH_SECRET_KEY = os.getenv("AUTH_SECRET_KEY", "secret_key")
AUTH_ALGORITHM = os.getenv("AUTH_ALGORITHM", "HS256")
AUTH_ACCESS_TOKEN_EXPIRE_MINUTES = int(
    os.getenv("AUTH_ACCESS_TOKEN_EXPIRE_MINUTES", 30)
)
AUTH_REFRESH_TOKEN_EXPIRE_DAYS = int(os.getenv("AUTH_REFRESH_TOKEN_EXPIRE_DAYS", 7))

# Email configuration
EMAIL_HOST = os.getenv("EMAIL_HOST", "smtp.gmail.com")
EMAIL_PORT = int(os.getenv("EMAIL_PORT", 587))
EMAIL_HOST_USER = os.getenv("EMAIL_HOST_USER", "your_email@gmail.com")
EMAIL_HOST_PASSWORD = os.getenv("EMAIL_HOST_PASSWORD", "your_password")
EMAIL_USE_TLS = os.getenv("EMAIL_USE_TLS", "True").lower() == "true"

# Logging configuration
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
LOG_FILE = os.getenv("LOG_FILE", "banking_network.log")
