import os
from dotenv import load_dotenv

load_dotenv()

DATABASE_URL = os.getenv("DATABASE_URL")
SECRET_KEY = os.getenv("SECRET_KEY")

def main():
    print(f"Database URL: {DATABASE_URL}")
    print(f"Secret Key: {SECRET_KEY}")

if __name__ == "__main__":
    main()
