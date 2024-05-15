# Improved code with async/await and logging
import logging
import asyncio
from pi_nexus.services.bank_api import fetch_bank_data

logging.basicConfig(level=logging.INFO)

async def main():
    async with aiohttp.ClientSession() as session:
        account_number = "1234567890"
        bank_data = await fetch_bank_data(session, account_number)
        logging.info(f"Fetched bank data: {bank_data}")

if __name__ == "__main__":
    asyncio.run(main())
