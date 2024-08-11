// transaction.rs

use std::collections::HashSet;
use std::hash::{Hash, Hasher};

// Transaction struct
pub struct Transaction {
    sender: String,
    recipient: String,
    amount: u64,
    timestamp: u64,
    hash: String,
}

impl Transaction {
    pub fn new(sender: String, recipient: String, amount: u64) -> Self {
        let timestamp = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_secs();
        let hash = Self::calculate_hash(sender.clone(), recipient.clone(), amount, timestamp);
        Transaction {
            sender,
            recipient,
            amount,
            timestamp,
            hash,
        }
    }

    fn calculate_hash(sender: String, recipient: String, amount: u64, timestamp: u64) -> String {
        let mut hasher = sha3::Sha3_256::new();
        hasher.write(sender.as_bytes());
        hasher.write(recipient.as_bytes());
        hasher.write(amount.to_le_bytes().as_ref());
        hasher.write(timestamp.to_le_bytes().as_ref());
        let hash = hasher.finish();
        hex::encode(hash)
    }

    pub fn hash(&self) -> String {
        self.hash.clone()
    }
}

// TransactionPool struct
pub struct TransactionPool {
    transactions: HashSet<Transaction>,
}

impl TransactionPool {
    pub fn new() -> Self {
        TransactionPool {
            transactions: HashSet::new(),
        }
    }

    pub fn add_transaction(&mut self, transaction: Transaction) {
        self.transactions.insert(transaction);
    }

    pub fn get_transactions(&self) -> Vec<Transaction> {
        self.transactions.clone().into_iter().collect()
    }
}
