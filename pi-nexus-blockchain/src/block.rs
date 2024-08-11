// block.rs

use std::collections::HashSet;
use std::hash::{Hash, Hasher};

use crate::transaction::Transaction;

// BlockHeader struct
pub struct BlockHeader {
    index: u32,
    previous_hash: String,
    transactions: Vec<Transaction>,
    timestamp: u64,
    nonce: u64,
    difficulty_target: u64,
}

impl BlockHeader {
    pub fn new(
        index: u32,
        previous_hash: String,
        transactions: Vec<Transaction>,
        difficulty_target: u64,
    ) -> Self {
        BlockHeader {
            index,
            previous_hash,
            transactions,
            timestamp: 0, // set timestamp later
            nonce: 0, // set nonce later
            difficulty_target,
        }
    }

    pub fn hash(&self) -> String {
        let mut hasher = sha3::Sha3_256::new();
        hasher.write(self.index.to_le_bytes().as_ref());
        hasher.write(self.previous_hash.as_bytes());
        for transaction in &self.transactions {
            hasher.write(transaction.hash().as_bytes());
        }
        hasher.write(self.timestamp.to_le_bytes().as_ref());
        hasher.write(self.nonce.to_le_bytes().as_ref());
        let hash = hasher.finish();
        hex::encode(hash)
    }
}

// Block struct
pub struct Block {
    header: BlockHeader,
    transactions: Vec<Transaction>,
}

impl Block {
    pub fn new(header: BlockHeader, transactions: Vec<Transaction>) -> Self {
        Block { header, transactions }
    }

    pub fn new_genesis() -> Self {
        let header = BlockHeader::new(0, "0".to_string(), vec![], 1000);
        Block { header, transactions: vec![] }
    }

    pub fn hash(&self) -> String {
        self.header.hash()
    }
}
