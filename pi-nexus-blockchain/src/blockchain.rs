// blockchain.rs (new)

use crate::block::Block;
use std::collections::VecDeque;

pub struct Blockchain {
    pub chain: VecDeque<Block>,
}

impl Blockchain {
    pub fn new() -> Self {
        let genesis_block = Block::new(0, "0", vec![]);
        Blockchain {
            chain: VecDeque::from(vec![genesis_block]),
        }
    }

    pub fn add_block(&mut self, transactions: Vec<crate::transaction::Transaction>) {
        let previous_hash = self.chain.back().unwrap().hash.clone();
        let new_block = Block::new(self.chain.len() as u64, &previous_hash, transactions);
        self.chain.push_back(new_block);
    }
}
