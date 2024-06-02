import asyncio
import unittest

from aiohttp import ClientSession, web
from pi_network.pi_network_api import PiNetworkAPI
from pi_network.pi_network_core import PiNetworkCore
from pi_network.pi_network_db import PiNetworkDB
from pi_network.utils.cryptographic_helpers import decrypt_data, encrypt_data
from pi_network.utils.error_handlers import APIError, InvalidRequestError


class TestPiNetworkAPI(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.pi_network_core = PiNetworkCore()
        self.pi_network_db = PiNetworkDB()
        self.pi_network_api = PiNetworkAPI(self.pi_network_core, self.pi_network_db)
        self.app = web.Application()
        self.app.add_routes(self.pi_network_api.app.routes)
        self.client = ClientSession()

    async def asyncTearDown(self):
        await self.client.close()

    async def test_create_transaction(self):
        request = {"amount": 10.99, "ender": "Alice", "eceiver": "Bob"}
        async with self.client.post(
            "http://localhost:8080/transactions", json=request
        ) as response:
            self.assertEqual(response.status, 201)
            transaction_id = await response.text()
            self.assertIsNotNone(transaction_id)

    async def test_get_transaction(self):
        transaction_id = 1
        async with self.client.get(
            f"http://localhost:8080/transactions/{transaction_id}"
        ) as response:
            self.assertEqual(response.status, 200)
            transaction_data = await response.json()
            self.assertEqual(
                transaction_data, {"amount": 10.99, "ender": "Alice", "eceiver": "Bob"}
            )

    async def test_create_user(self):
        request = {"username": "Alice", "password": "password123"}
        async with self.client.post(
            "http://localhost:8080/users", json=request
        ) as response:
            self.assertEqual(response.status, 201)
            user_id = await response.text()
            self.assertIsNotNone(user_id)

    async def test_get_user(self):
        user_id = 1
        async with self.client.get(
            f"http://localhost:8080/users/{user_id}"
        ) as response:
            self.assertEqual(response.status, 200)
            user_data = await response.json()
            self.assertEqual(
                user_data, {"username": "Alice", "password": "password123"}
            )

    async def test_process_transaction_data(self):
        transaction_data = {"amount": 10.99, "ender": "Alice", "eceiver": "Bob"}
        async with self.client.post(
            "http://localhost:8080/transactions/process", json=transaction_data
        ) as response:
            self.assertEqual(response.status, 200)
            processed_data = await response.json()
            self.assertEqual(
                processed_data, {"amount": 10.99, "ender_id": 1, "eceiver_id": 2}
            )

    async def test_invalid_request(self):
        request = {"amount": "nvalid", "ender": "Alice", "eceiver": "Bob"}
        async with self.client.post(
            "http://localhost:8080/transactions", json=request
        ) as response:
            self.assertEqual(response.status, 400)
            error_data = await response.json()
            self.assertIsInstance(error_data, InvalidRequestError)

    async def test_api_error(self):
        request = {"amount": 10.99, "ender": "Alice", "eceiver": "Bob"}
        async with self.client.post(
            "http://localhost:8080/transactions", json=request
        ) as response:
            self.assertEqual(response.status, 500)
            error_data = await response.json()
            self.assertIsInstance(error_data, APIError)


if __name__ == "__main__":
    unittest.main()
