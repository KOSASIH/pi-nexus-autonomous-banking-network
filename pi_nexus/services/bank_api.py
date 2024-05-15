# Improved code with async/await and error handling
import aiohttp
from aiohttp import ClientError

async def fetch_bank_data(session: aiohttp.ClientSession, account_number: str) -> dict:
    try:
        async with session.get(f"{Constants.BANK_API_URL}/{account_number}") as response:
            response.raise_for_status()
            return await response.json()
    except ClientError as e:
        print(f"Error fetching bank data: {e}")
        return {}
