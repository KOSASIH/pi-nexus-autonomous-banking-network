import hashlib
import random
import time
from typing import List

class ProofOfStake:
    def __init__(self, validators: List[str], block_time: int = 10):
        self.validators = validators
        self.block_time = block_time
        self.stakes = {}
        self.current_block_hash = None

    def add_stake(self, validator: str, stake: int):
        if validator not in self.validators:
            raise ValueError("Validator not found")
        self.stakes[validator] = stake

    def remove_stake(self, validator: str):
        if validator not in self.stakes:
            raise ValueError("Validator not found")
        del self.stakes[validator]

    def get_next_block_hash(self):
        if not self.current_block_hash:
            return self.generate_genesis_block_hash()
        return self.generate_block_hash(self.current_block_hash)

    def generate_genesis_block_hash(self):
        genesis_block_hash = hashlib.sha256("Genesis Block".encode()).hexdigest()
        self.current_block_hash = genesis_block_hash
        return genesis_block_hash

    def generate_block_hash(self, previous_block_hash: str):
        timestamp = int(time.time())
        block_hash = hashlib.sha256(f"{previous_block_hash}{timestamp}".encode()).hexdigest()
        self.current_block_hash = block_hash
        return block_hash

    def select_validator(self):
        total_stake = sum(self.stakes.values())
        selection = random.randint(0, total_stake - 1)
        cumulative_stake = 0
        for validator, stake in self.stakes.items():
            cumulative_stake += stake
            if selection < cumulative_stake:
                return validator
        return None

    def run_consensus(self):
        while True:
            next_block_hash = self.get_next_block_hash()
            selected_validator = self.select_validator()
            if selected_validator:
                print(f"Selected validator: {selected_validator}")
                print(f"Next block hash: {next_block_hash}")
                # Add block to blockchain
                time.sleep(self.block_time)
            else:
                print("No validator selected")
                time.sleep(self.block_time)
