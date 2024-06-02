import asyncio
import json
import logging
from aiohttp import web
from pi_network.pi_network_core import PiNetworkCore
from pi_network.utils.cryptographic_helpers import encrypt_data, decrypt_data
from pi_network.utils.data_processing_helpers import process_transaction_data

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class PiNetworkAPI:
    def __init__(self, pi_network_core: PiNetworkCore, db: 'PiNetworkDB'):
        self.pi_network_core = pi_network_core
        self.db = db
        self.app = web.Application()
        self.app.add_routes([
            web.post('/transactions', self.create_transaction),
            web.get('/transactions/{transaction_id}', self.get_transaction),
            web.post('/users', self.create_user),
            web.get('/users/{user_id}', self.get_user),
        ])

    async def create_transaction(self, request: web.Request):
        try:
            data = await request.json()
            transaction_data = process_transaction_data(data)
            encrypted_data = encrypt_data(transaction_data)
            transaction_id = await self.pi_network_core.create_transaction(encrypted_data)
            return web.Response(text=json.dumps({'transaction_id': transaction_id}), status=201)
        except Exception as e:
            logger.error(f"Error creating transaction: {e}")
            return web.Response(text=json.dumps({'error': str(e)}), status=500)

    async def get_transaction(self, request: web.Request):
        try:
            transaction_id = request.match_info['transaction_id']
            transaction_data = await self.pi_network_core.get_transaction(transaction_id)
            decrypted_data = decrypt_data(transaction_data)
            return web.Response(text=json.dumps(decrypted_data), status=200)
        except Exception as e:
            logger.error(f"Error getting transaction: {e}")
            return web.Response(text=json.dumps({'error': str(e)}), status=404)

    async def create_user(self, request: web.Request):
        try:
            data = await request.json()
            user_id = await self.pi_network_core.create_user(data)
            return web.Response(text=json.dumps({'user_id': user_id}), status=201)
        except Exception as e:
            logger.error(f"Error creating user: {e}")
            return web.Response(text=json.dumps({'error': str(e)}), status=500)

    async def get_user(self, request: web.Request):
        try:
            user_id = request.match_info['user_id']
            user_data = await self.pi_network_core.get_user(user_id)
            return web.Response(text=json.dumps(user_data), status=200)
        except Exception as e:
            logger.error(f"Error getting user: {e}")
            return web.Response(text=json.dumps({'error': str(e)}), status=404)

    def run(self):
        web.run_app(self.app, port=8080)

if __name__ == '__main__':
    pi_network_core = PiNetworkCore()
    db = PiNetworkDB()
    api = PiNetworkAPI(pi_network_core, db)
    api.run()
