import asyncio
from web3 import Web3
from pi_bridge_contract import PiBridgeContract

w3 = Web3(Web3.HTTPProvider("https://mainnet.infura.io/v3/YOUR_PROJECT_ID"))
pi_bridge_contract = PiBridgeContract("0x...PiBridgeContractAddress...")

async def event_listener():
    async for event in pi_bridge_contract.events.Deposit.create_filter(fromBlock="latest"):
        print(f"Deposit event: {event.args.user} deposited {event.args.amount} Pi tokens")
        # Perform additional logic or notifications here

    async for event in pi_bridge_contract.events.Withdrawal.create_filter(fromBlock="latest"):
        print(f"Withdrawal event: {event.args.user} withdrew {event.args.amount} Pi tokens")
        # Perform additional logic or notifications here

asyncio.run(event_listener())
