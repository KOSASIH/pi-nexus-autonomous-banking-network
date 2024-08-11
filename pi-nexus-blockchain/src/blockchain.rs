// blockchain.rs

use std::collections::{HashMap, VecDeque};
use std::hash::{Hash, Hasher};
use std::sync::{Arc, Mutex};

use crate::block::{Block, BlockHeader};
use crate::transaction::{Transaction, TransactionPool};

// Blockchain struct
pub struct Blockchain {
    chain: Vec<Block>,
    transaction_pool: TransactionPool,
    pending_transactions: VecDeque<Transaction>,
    block_time: u64,
    difficulty_target: u64,
    reward: u64,
    node_id: String,
}

impl Blockchain {
    pub fn new(node_id: String) -> Self {
        let genesis_block = Block::new_genesis();
        let transaction_pool = TransactionPool::new();
        Blockchain {
            chain: vec![genesis_block],
            transaction_pool,
            pending_transactions: VecDeque::new(),
            block_time: 10, // 10 seconds
            difficulty_target: 1000, // adjust difficulty target
            reward: 100, // block reward
            node_id,
        }
    }

    pub fn add_transaction(&mut self, transaction: Transaction) {
        self.pending_transactions.push_back(transaction);
    }

    pub fn mine_block(&mut self) {
        let block = self.create_block();
        self.chain.push(block);
        self.pending_transactions.clear();
    }

    fn create_block(&mut self) -> Block {
        let previous_block_hash = self.chain.last().unwrap().hash.clone();
        let transactions = self.pending_transactions.clone();
        let block_header = BlockHeader::new(
            self.chain.len() as u32,
            previous_block_hash,
            transactions.clone(),
            self.difficulty_target,
        );
        let block = Block::new(block_header, transactions);
        block
    }

    pub fn get_chain(&self) -> Vec<Block> {
        self.chain.clone()
    }

    pub fn get_transaction_pool(&self) -> TransactionPool {
        self.transaction_pool.clone()
    }
}

// Blockchain node
pub struct BlockchainNode {
    blockchain: Arc<Mutex<Blockchain>>,
}

impl BlockchainNode {
    pub fn new(node_id: String) -> Self {
        let blockchain = Arc::new(Mutex::new(Blockchain::new(node_id)));
        BlockchainNode { blockchain }
    }

    pub fn add_transaction(&self, transaction: Transaction) {
        self.blockchain.lock().unwrap().add_transaction(transaction);
    }

    pub fn mine_block(&self) {
        self.blockchain.lock().unwrap().mine_block();
    }

    pub fn get_chain(&self) -> Vec<Block> {
        self.blockchain.lock().unwrap().get_chain()
    }

    pub fn get_transaction_pool(&self) -> TransactionPool {
        self.blockchain.lock().unwrap().get_transaction_pool()
    }
}
