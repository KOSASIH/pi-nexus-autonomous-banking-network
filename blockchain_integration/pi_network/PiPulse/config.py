import os

# Configuration settings
class Config:
    # API settings
    API_HOST = '0.0.0.0'
    API_PORT = 5000
    API_DEBUG = True

    # Database settings
    DB_HOST = 'localhost'
    DB_PORT = 5432
    DB_USERNAME = 'pipulse'
    DB_PASSWORD = 'pipulse'
    DB_NAME = 'pipulse'

    # Security settings
    SECRET_KEY = 'super_secret_key_here'
    JWT_SECRET_KEY = 'super_secret_jwt_key_here'
    JWT_TOKEN_EXPIRATION = 3600  # 1 hour

    # Email settings
    EMAIL_HOST = 'smtp.gmail.com'
    EMAIL_PORT = 587
    EMAIL_USERNAME = 'pipulse@example.com'
    EMAIL_PASSWORD = 'pipulse_password'

    # File storage settings
    UPLOAD_FOLDER = 'uploads'
    ALLOWED_EXTENSIONS = {'txt', 'pdf', 'png', 'jpg', 'jpeg', 'gif'}

    # API keys
    OPENWEATHERMAP_API_KEY = 'your_openweathermap_api_key_here'

# Load environment variables from .env file
if os.path.exists('.env'):
    with open('.env', 'r') as f:
        for line in f:
            key, value = line.strip().split('=')
            os.environ[key] = value

# Create configuration object
config = Config()
