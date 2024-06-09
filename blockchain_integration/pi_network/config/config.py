import os

# Environment variables
ENV = os.environ.get('ENV', 'development')

# Blockchain settings
BLOCKCHAIN_NETWORK = os.environ.get('BLOCKCHAIN_NETWORK', 'ainnet')
BLOCKCHAIN_RPC_URL = os.environ.get('BLOCKCHAIN_RPC_URL', 'https://mainnet.pi.network/rpc')
BLOCKCHAIN_CHAIN_ID = os.environ.get('BLOCKCHAIN_CHAIN_ID', 1)

# PI Network settings
PI_NETWORK_API_URL = os.environ.get('PI_NETWORK_API_URL', 'https://api.pi.network')
PI_NETWORK_API_KEY = os.environ.get('PI_NETWORK_API_KEY', '')

# Database settings
DB_HOST = os.environ.get('DB_HOST', 'localhost')
DB_PORT = int(os.environ.get('DB_PORT', 5432))
DB_USERNAME = os.environ.get('DB_USERNAME', 'pi_network')
DB_PASSWORD = os.environ.get('DB_PASSWORD', '')
DB_NAME = os.environ.get('DB_NAME', 'pi_network')

# Logging settings
LOG_LEVEL = os.environ.get('LOG_LEVEL', 'INFO')
LOG_FILE = os.environ.get('LOG_FILE', 'pi_network.log')

# Other settings
NODE_URL = os.environ.get('NODE_URL', 'https://node.pi.network')
NODE_PORT = int(os.environ.get('NODE_PORT', 8080))

# Development settings (override production settings)
if ENV == 'development':
    BLOCKCHAIN_RPC_URL = 'https://testnet.pi.network/rpc'
    PI_NETWORK_API_URL = 'https://api.test.pi.network'
    DB_HOST = 'localhost'
    DB_PORT = 5433
    DB_USERNAME = 'pi_network_dev'
    DB_PASSWORD = ''
    DB_NAME = 'pi_network_dev'
    LOG_LEVEL = 'DEBUG'
    LOG_FILE = 'pi_network_dev.log'
