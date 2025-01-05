"""
Pi Coin Configuration Constants
This module contains constants related to the Pi Coin cryptocurrency.
"""

# Pi Coin Symbol
PI_COIN_SYMBOL = "Pi"  # Symbol for Pi Coin

# Pi Coin Value
PI_COIN_VALUE = 314159  # Fixed value of Pi Coin in USD

# Pi Coin Supply
PI_COIN_SUPPLY = 100_000_000_000  # Total supply of Pi Coin
PI_COIN_DYNAMIC_SUPPLY = True  # Enable dynamic supply adjustments based on demand

# Pi Coin Transaction Fee
PI_COIN_TRANSACTION_FEE = 0.01  # Transaction fee in USD
PI_COIN_TRANSACTION_FEE_ADJUSTMENT = 0.001  # Dynamic adjustment factor for transaction fees

# Pi Coin Block Time
PI_COIN_BLOCK_TIME = 10  # Average block time in seconds
PI_COIN_BLOCK_TIME_ADJUSTMENT = 1  # Adjustment factor for block time based on network load

# Pi Coin Mining Difficulty
PI_COIN_MINING_DIFFICULTY = 1000  # Difficulty level for mining Pi Coin
PI_COIN_MINING_DIFFICULTY_ADJUSTMENT = 0.1  # Adjustment factor for mining difficulty

# Pi Coin Reward for Mining
PI_COIN_MINING_REWARD = 12.5  # Reward for mining a block
PI_COIN_MINING_REWARD_ADJUSTMENT = 0.5  # Dynamic adjustment for mining rewards

# Pi Coin Network Protocol
PI_COIN_NETWORK_PROTOCOL = "PoS"  # Proof of Stake
PI_COIN_NETWORK_PROTOCOL_VERSION = "1.0.0"  # Version of the network protocol

# Pi Coin Maximum Transaction Size
PI_COIN_MAX_TRANSACTION_SIZE = 1_000_000  # Maximum transaction size in bytes

# Pi Coin Decimals
PI_COIN_DECIMALS = 18  # Number of decimal places for Pi Coin

# Pi Coin Genesis Block Timestamp
PI_COIN_GENESIS_BLOCK_TIMESTAMP = "2025-01-01T00:00:00Z"  # Timestamp of the genesis block

# Pi Coin Governance Model
PI_COIN_GOVERNANCE_MODEL = "Decentralized"  # Governance model for Pi Coin
PI_COIN_GOVERNANCE_VOTING_PERIOD = 604800  # Voting period in seconds (1 week)

# Pi Coin Security Features
PI_COIN_ENCRYPTION_ALGORITHM = "AES-256"  # Encryption algorithm for securing transactions
PI_COIN_HASHING_ALGORITHM = "SHA-512"  # Enhanced hashing algorithm for block verification
PI_COIN_SIGNATURE_SCHEME = "ECDSA"  # Digital signature scheme for transaction signing
PI_COIN_SECURITY_AUDIT_INTERVAL = 86400  # Security audit interval in seconds (1 day)

# Pi Coin Network Parameters
PI_COIN_MAX_PEERS = 100  # Maximum number of peers in the network
PI_COIN_NODE_TIMEOUT = 30  # Timeout for node responses in seconds
PI_COIN_CONNECTION_RETRY_INTERVAL = 5  # Retry interval for node connections in seconds

# Pi Coin Staking Parameters
PI_COIN_MIN_STAKE_AMOUNT = 100  # Minimum amount required to stake
PI_COIN_STAKE_REWARD_RATE = 0.05  # Annual reward rate for staking
PI_COIN_STAKE_LOCK_PERIOD = 2592000  # Lock period for staked coins in seconds (30 days)

# Pi Coin API Rate Limits
PI_COIN_API_REQUEST_LIMIT = 1000  # Maximum API requests per hour
PI_COIN_API_KEY_EXPIRATION = 3600  # API key expiration time in seconds

# Pi Coin Regulatory Compliance
PI_COIN_KYC_REQUIRED = True  # Whether KYC is required for transactions
PI_COIN_COMPLIANCE_JURISDICTIONS = ["US", "EU", "UK", "SG", "JP"]  # Expanded jurisdictions for compliance

# Additional constants can be added here as needed
