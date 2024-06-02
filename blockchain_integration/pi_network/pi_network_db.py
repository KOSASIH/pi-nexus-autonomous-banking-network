import asyncio

import aiomysql
from pi_network.utils.cryptographic_helpers import decrypt_data, encrypt_data


class PiNetworkDB:
    def __init__(self, host: str, user: str, password: str, db: str):
        self.host = host
        self.user = user
        self.password = password
        self.db = db
        self.pool = None

    async def connect(self):
        self.pool = await aiomysql.create_pool(
            host=self.host, user=self.user, password=self.password, db=self.db
        )

    async def create_transaction(self, encrypted_data: bytes):
        async with self.pool.acquire() as conn:
            async with conn.cursor() as cur:
                await cur.execute(
                    "INSERT INTO transactions (data) VALUES (%s)", (encrypted_data,)
                )
                transaction_id = cur.lastrowid
                await conn.commit()
                return transaction_id

    async def get_transaction(self, transaction_id: int):
        async with self.pool.acquire() as conn:
            async with conn.cursor() as cur:
                await cur.execute(
                    "SELECT data FROM transactions WHERE id = %s", (transaction_id,)
                )
                row = await cur.fetchone()
                if row:
                    encrypted_data = row[0]
                    decrypted_data = decrypt_data(encrypted_data)
                    return decrypted_data
                else:
                    return None

    async def create_user(self, user_data: dict):
        async with self.pool.acquire() as conn:
            async with conn.cursor() as cur:
                await cur.execute(
                    "INSERT INTO users (data) VALUES (%s)", (json.dumps(user_data),)
                )
                user_id = cur.lastrowid
                await conn.commit()
                return user_id

    async def get_user(self, user_id: int):
        async with self.pool.acquire() as conn:
            async with conn.cursor() as cur:
                await cur.execute("SELECT data FROM users WHERE id = %s", (user_id,))
                row = await cur.fetchone()
                if row:
                    user_data = json.loads(row[0])
                    return user_data
                else:
                    return None


if __name__ == "__main__":
    db = PiNetworkDB(
        host="localhost", user="root", password="password", db="pi_network"
    )
    asyncio.run(db.connect())
