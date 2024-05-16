# config.py
from decouple import config

API_KEY = config("FANTASY_EXCHANGE_API_KEY")
DATABASE_URL = config("DATABASE_URL")
