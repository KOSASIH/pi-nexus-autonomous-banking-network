// block.rs (update)

use crate::transaction::Transaction;

pub struct Block {
    pub index: u64,
    pub previous_hash: String,
    pub timestamp: u64,
    pub transactions: Vec<Transaction>,
    pub hash: String,
}

impl Block {
    pub fn new(index: u64, previous_hash: &str, transactions: Vec<Transaction>) -> Self {
        let timestamp = crate::utils::get_current_timestamp();
        let hash = crate::utils::calculate_block_hash(index, previous_hash, &transactions, timestamp);
        Block {
            index,
            previous_hash: previous_hash.to_string(),
            timestamp,
            transactions,
            hash,
        }
    }
}
