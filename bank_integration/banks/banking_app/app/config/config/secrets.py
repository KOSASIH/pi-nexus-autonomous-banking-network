import os

class Secrets:
    def __init__(self):
        self.secret_key = os.environ.get("SECRET_KEY")
        self.jwt_secret_key = os.environ.get("JWT_SECRET_KEY")
        self.database_url = os.environ.get("DATABASE_URL")

    def get_secret_key(self):
        return self.secret_key

    def get_jwt_secret_key(self):
        return self.jwt_secret_key

    def get_database_url(self):
        return self.database_url
