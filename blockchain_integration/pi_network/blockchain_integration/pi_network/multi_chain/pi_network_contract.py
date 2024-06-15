import asyncio
from web3 import Web3

class PINetworkContract:
    def __init__(self, contract_address, abi):
        self.contract_address = contract_address
        self.abi = abi
        self.w3 = Web3(Web3.HTTPProvider('https://mainnet.infura.io/v3/YOUR_PROJECT_ID'))

    async def get_user_balance(self, user_address):
        contract = self.w3.eth.contract(address=self.contract_address, abi=self.abi)
        balance = contract.functions.balanceOf(user_address).call()
        return balance

    async def transfer_funds(self, from_address, to_address, amount):
        contract = self.w3.eth.contract(address=self.contract_address, abi=self.abi)
        tx = contract.functions.transfer(to_address, amount).buildTransaction({
            'from': from_address,
            'gas': 20000,
            'gasPrice': self.w3.eth.gas_price
        })
        self.w3.eth.send_transaction(tx)

    async def run(self):
        while True:
            # Example usage:
            balance = await self.get_user_balance('0x742d35Cc6634C0532925a3b844Bc454e4438f44e')
            print(f'User Balance: {balance}')
            await asyncio.sleep(10)

if __name__ == '__main__':
    contract = PINetworkContract('0x742d35Cc6634C0532925a3b844Bc454e4438f44e', [])
    asyncio.run(contract.run())
