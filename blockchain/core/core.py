from .block import Block
from .transaction import Transaction, TransactionPool
from .wallet import Wallet, WalletPool

def create_genesis_block():
    return Block(0, datetime.now(), "Genesis Block", "0", "0")

def create_transaction_pool():
    return TransactionPool()

def create_wallet_pool():
    return WalletPool()

def create_wallet():
    public_key, private_key = generate_keys()
    return Wallet(public_key, private_key)

def generate_keys():
    import rsa
    (public, private) = rsa.newkeys(512)
    return (public.save_pkcs1().decode(), private.save_pkcs1().decode())

def add_transaction_to_block(block: Block, transaction: Transaction):
    block.data.append(transaction)

def calculate_block_hash(block: Block):
    return Block.calculate_hash(block.index, block.timestamp, block.data, block.previous_hash)

def add_block_to_chain(chain: List[Block], block: Block):
    chain.append(block)

def get_balance(wallet_pool: WalletPool, public_key: str):
    wallet = wallet_pool.get_wallet(public_key)
    if wallet:
        balance = 0
        for transaction in wallet.transactions:
            if transaction.sender == public_key:
                balance -= transaction.amount
            elif transaction.receiver == public_key:
                balance += transaction.amount
        return balance
    return 0

def create_blockchain():
    genesis_block = create_genesis_block()
    blockchain = [genesis_block]
    return blockchain

def add_block_to_blockchain(blockchain: List[Block], block: Block):
    add_block_to_chain(blockchain, block)

def get_blockchain_length(blockchain: List[Block]):
    return len(blockchain)

def get_block_by_index(blockchain: List[Block], index: int):
    if index < 0 or index >= get_blockchain_length(blockchain):
        return None
    return blockchain[index]

def get_block_by_hash(blockchain: List[Block], hash: str):
    for block in blockchain:
        if block.hash == hash:
            return block
    return None

def validate_blockchain(blockchain: List[Block]):
    for i in range(1, get_blockchain_length(blockchain)):
        block = blockchain[i]
        previous_block = blockchain[i - 1]
        if block.previous_hash != previous_block.hash:
            return False
        if block.hash != calculate_block_hash(block):
            return False
    return True
