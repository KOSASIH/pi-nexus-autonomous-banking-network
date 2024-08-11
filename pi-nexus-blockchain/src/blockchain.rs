// blockchain.rs (update)

use crate::block::Block;
use crate::transaction::Transaction;
use std::collections::VecDeque;

pub struct Blockchain {
    pub chain: VecDeque<Block>,
    pub transactions: Vec<Transaction>,
}

impl Blockchain {
    pub fn new() -> Self {
        let genesis_block = Block::new(0, "0", vec![]);
        Blockchain {
            chain: VecDeque::from(vec![genesis_block]),
            transactions: vec![],
        }
    }

    pub fn add_transaction(&mut self, transaction: Transaction) {
        self.transactions.push(transaction);
    }

    pub fn add_block(&mut self) {
        let transactions = self.transactions.clone();
        self.transactions.clear();
        let previous_hash = self.chain.back().unwrap().hash.clone();
        let new_block = Block::new(self.chain.len() as u64, &previous_hash, transactions);
        self.chain.push_back(new_block);
    }

    pub fn get_transactions(&self) -> Vec<Transaction> {
        self.transactions.clone()
    }
}
