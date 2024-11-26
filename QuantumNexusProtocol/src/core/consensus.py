import hashlib
import time
from collections import defaultdict

class Block:
    def __init__(self, index, previous_hash, transactions, timestamp=None):
        self.index = index
        self.previous_hash = previous_hash
        self.transactions = transactions
        self.timestamp = timestamp or time.time()
        self.hash = self.calculate_hash()

    def calculate_hash(self):
        block_string = f"{self.index}{self.previous_hash}{self.transactions}{self.timestamp}".encode()
        return hashlib.sha256(block_string).hexdigest()

class ConsensusAlgorithm:
    def __init__(self):
        self.validators = {}
        self.current_block = None
        self.pending_transactions = []
        self.validator_rewards = defaultdict(int)
        self.slashing_conditions = {}

    def register_validator(self, validator_id, stake):
        self.validators[validator_id] = stake

    def propose_block(self, transactions):
        if not self.validators:
            raise Exception("No validators registered.")
        
        self.current_block = Block(
            index=len(self.pending_transactions) + 1,
            previous_hash=self.get_last_block_hash(),
            transactions=transactions
        )
        return self.current_block

    def validate_block(self, block):
        # Validate block hash and structure
        if block.hash != block.calculate_hash():
            return False
        if block.previous_hash != self.get_last_block_hash():
            return False
        return True

    def finalize_block(self, block):
        if self.validate_block(block):
            self.pending_transactions.append(block)
            self.reward_validators(block)
            return True
        return False

    def reward_validators(self, block):
        # Reward validators for proposing a valid block
        for validator in self.validators:
            self.validator_rewards[validator] += 1  # Simple reward mechanism

    def get_last_block_hash(self):
        return self.pending_transactions[-1].hash if self.pending_transactions else "0"

    def handle_fork(self, competing_chain):
        # Implement logic to handle forks in the blockchain
        if len(competing_chain) > len(self.pending_transactions):
            self.pending_transactions = competing_chain

    def slash_validator(self, validator_id):
        # Implement slashing conditions for malicious behavior
        if validator_id in self.slashing_conditions:
            # Deduct stake or impose penalties
            self.validators[validator_id] = max(0, self.validators[validator_id] - self.slashing_conditions[validator_id])

    def set_slashing_condition(self, validator_id, penalty):
        self.slashing_conditions[validator_id] = penalty

    def get_validator_rewards(self):
        return self.validator_rewards

    def get_pending_transactions(self):
        return self.pending_transactions

    def get_validators(self):
        return self.validators
