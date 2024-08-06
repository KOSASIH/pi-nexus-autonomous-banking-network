import hashlib
import random
import time
from typing import List

class DelegatedProofOfStake:
    def __init__(self, validators: List[str], block_time: int = 10):
        self.validators = validators
        self.block_time = block_time
        self.votes = {}
        self.delegates = {}
        self.current_block_hash = None

    def add_vote(self, voter: str, delegate: str):
        if voter not in self.validators:
            raise ValueError("Voter not found")
        if delegate not in self.validators:
            raise ValueError("Delegate not found")
        self.votes[voter] = delegate

    def remove_vote(self, voter: str):
        if voter not in self.votes:
            raise ValueError("Voter not found")
        del self.votes[voter]

    def add_delegate(self, delegate: str):
        if delegate not in self.validators:
            raise ValueError("Delegate not found")
        self.delegates[delegate] = 0

    def remove_delegate(self, delegate: str):
        if delegate not in self.delegates:
            raise ValueError("Delegate not found")
        del self.delegates[delegate]

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

    def select_delegate(self):
        total_votes = sum(self.delegates.values())
        selection = random.randint(0, total_votes - 1)
        cumulative_votes = 0
        for delegate, votes in self.delegates.items():
            cumulative_votes += votes
            if selection < cumulative_votes:
                return delegate
        return None

    def run_consensus(self):
        while True:
            next_block_hash = self.get_next_block_hash()
            selected_delegate = self.select_delegate()
            if selected_delegate:
                print(f"Selected delegate: {selected_delegate}")
                print(f"Next block hash: {next_block_hash}")
                # Add block to blockchain
                time.sleep(self.block_time)
            else:
                print("No delegate selected")
                time.sleep(self.block_time)
