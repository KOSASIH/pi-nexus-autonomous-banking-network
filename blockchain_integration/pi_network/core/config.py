class Config:
    SECRET_KEY = "secret_key_here"
    SQLALCHEMY_DATABASE_URI = "postgresql://user:password@localhost/pi_nexus_db"
    BLOCKCHAIN_NODE_URL = "https://mainnet.infura.io/v3/YOUR_PROJECT_ID"
    AI_MODEL_PATH = "pi_network/ai_analytics/models/transaction_prediction.pkl"
