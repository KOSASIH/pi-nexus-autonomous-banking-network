import os
import json
import hashlib
import time
from datetime import datetime
from typing import List, Dict, Any

class Blockchain:
    """
    The Blockchain class implements the core functionality of the Pi network's blockchain.
    """

    def __init__(self, pi_coin_symbol: str, stable_value: float, exchanges: List[str]):
        """
        Initializes a new Blockchain object.

        :param pi_coin_symbol: The symbol of the Pi coin.
        :param stable_value: The stable value of the Pi coin.
        :param exchanges: The list of exchanges where the Pi coin is listed.
        """

        self.pi_coin_symbol = pi_coin_symbol
        self.stable_value = stable_value
        self.exchanges = exchanges
        self.chain = []
        self.pending_transactions = []
        self.nodes = set()

        # Create the genesis block
        genesis_block = self.create_block(proof=0, previous_hash='0' * 64)
        self.chain.append(genesis_block)

    def create_block(self, proof: int, previous_hash: str) -> Dict[str, Any]:
        """
        Creates a new block and adds it to the blockchain.

        :param proof: The proof of work for the new block.
        :param previous_hash: The hash of the previous block.
        :return: The new block.
        """

        block = {
            'index': len(self.chain),
            'timestamp': int(time.time()),
            'transactions': self.pending_transactions,
            'proof': proof,
            'previous_hash': previous_hash,
            'pi_coin_symbol': self.pi_coin_symbol,
            'stable_value': self.stable_value,
            'exchanges': self.exchanges,
        }

        # Reset the pending transactions
        self.pending_transactions = []

        # Add the new block to the blockchain
        self.chain.append(block)

        return block

    def add_transaction(self, sender: str, recipient: str, amount: float) -> None:
        """
        Adds a new transaction to the pending transactions list.

        :param sender: The sender of the transaction.
        :param recipient: The recipient of the transaction.
        :param amount: The amount of the transaction.
        """

        self.pending_transactions.append({
            'sender': sender,
            'recipient': recipient,
            'amount': amount,
        })

    def proof_of_work(self, last_block: Dict[str, Any]) -> int:
        """
        Finds the proof of work for the new block.

        :param last_block: The last block in the blockchain.
        :return: The proof of work for the new block.
        """

        proof = 0
        while not self.is_valid_proof(last_block['proof'], proof):
            proof += 1

        return proof

    def is_valid_proof(self, last_block_proof: int, proof: int) -> bool:
        """
        Verifies the proof of work for a new block.

        :param last_block_proof: The proof of work of the last block.
        :param proof: The proof of work of the new block.
        :return: True if the proof is valid, False otherwise.
        """

        guess = f'{last_block_proof}{proof}'.encode()
        guess_hash = hashlib.sha256(guess).hexdigest()
        return guess_hash[:4] == '0000'

    def is_valid(self) -> bool:
        """
        Verifies the validity of the blockchain.

        :return: True if the blockchain is valid, False otherwise.
        """

        for i in range(1, len(self.chain)):
            current_block = self.chain[i]
            previous_block = self.chain[i - 1]

            if current_block['previous_hash'] != self.hash(previous_block):
                return False

            if not self.is_valid_proof(previous_block['proof'], current_block['proof']):
                return False

        return True

    def hash(self, block: Dict[str, Any]) -> str:
        """
        Hashes a block.

        :param block: The block to hash.
        :return: The hash of the block.
        """

        block_string = json.dumps(block, sort_keys=True).encode()
        return hashlib.sha256(block_string).hexdigest()

    def add_node(self, node: str) -> None:
        """
        Adds a new node to the blockchain.

        :param node: The address of the new node.
        """

        self.nodes.add(node)

    def replace_chain(self) -> None:
        """
        Replaces the current blockchain with the longest valid chain.
        """

        longest_chain = None
        max_length = len(self.chain)

        for node in self.nodes:
            response = requests.get(f'{node}/chain')

            if response.status_code == 200:
                length = response.json()['length']
                chain = response.json()['chain']

                if length > max_length and self.is_valid(chain):
                    longest_chain = chain
                    max_length = length

        if longest_chain:
            self.chain = longest_chain
