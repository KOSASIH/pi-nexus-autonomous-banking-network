// consensus.rs
use std::collections::{HashMap, VecDeque};
use std::sync::{Arc, Mutex};

use crate::chain::{Chain, ChainConfig};

pub struct ConsensusManager {
    consensus_algorithms: HashMap<String, ConsensusAlgorithm>,
    chain_configs: HashMap<String, ChainConfig>,
}

impl ConsensusManager {
    pub fn new() -> Self {
        ConsensusManager {
            consensus_algorithms: HashMap::new(),
            chain_configs: HashMap::new(),
        }
    }

    pub fn add_consensus_algorithm(&mut self, algorithm_id: String, algorithm: ConsensusAlgorithm) {
        self.consensus_algorithms.insert(algorithm_id, algorithm);
    }

    pub fn add_chain_config(&mut self, chain_id: String, chain_config: ChainConfig) {
        self.chain_configs.insert(chain_id, chain_config);
    }

    pub fn get_consensus_algorithm(&self, algorithm_id: &str) -> Option<&ConsensusAlgorithm> {
        self.consensus_algorithms.get(algorithm_id)
    }

    pub fn get_chain_config(&self, chain_id: &str) -> Option<&ChainConfig> {
        self.chain_configs.get(chain_id)
    }
}

pub trait ConsensusAlgorithm {
    fn validate_transaction(&self, transaction: &Transaction) -> bool;
    fn validate_block(&self, block: &Block) -> bool;
}

pub struct Transaction {
    transaction_id: String,
    sender: String,
    recipient: String,
    amount: u64,
    data: Vec<u8>,
}

impl Transaction {
    pub fn new(
        transaction_id: String,
        sender: String,
        recipient: String,
        amount: u64,
        data: Vec<u8>,
    ) -> Self {
        Transaction {
            transaction_id,
            sender,
            recipient,
            amount,
            data,
        }
    }

    pub fn get_transaction_id(&self) -> &str {
        &self.transaction_id
    }

    pub fn get_sender(&self) -> &str {
        &self.sender
    }

    pub fn get_recipient(&self) -> &str {
        &self.recipient
    }

    pub fn get_amount(&self) -> u64 {
        self.amount
    }

    pub fn get_data(&self) -> &[u8] {
        &self.data
    }
}

pub struct Block {
    block_id: String,
    transactions: Vec<Transaction>,
    previous_block_hash: String,
    timestamp: u64,
}

impl Block {
    pub fn new(
        block_id: String,
        transactions: Vec<Transaction>,
        previous_block_hash: String,
        timestamp: u64,
    ) -> Self {
        Block {
            block_id,
            transactions,
            previous_block_hash,
            timestamp,
        }
    }

    pub fn get_block_id(&self) -> &str {
        &self.block_id
    }

    pub fn get_transactions(&self) -> &[Transaction] {
        &self.transactions
    }

    pub fn get_previous_block_hash(&self) -> &str {
        &self.previous_block_hash
    }

    pub fn get_timestamp(&self) -> u64 {
        self.timestamp
    }
}
