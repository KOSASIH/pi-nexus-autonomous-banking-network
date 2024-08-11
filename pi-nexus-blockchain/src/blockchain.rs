// blockchain.rs (updated)

// ...

impl Blockchain {
    // ...

    pub fn validate_block(&self, block: &Block) -> bool {
        let previous_block_hash = self.chain.last().unwrap().hash.clone();
        if block.header.previous_hash != previous_block_hash {
            return false;
        }
        if block.header.difficulty_target != self.difficulty_target {
            return false;
        }
        if block.header.timestamp <= self.chain.last().unwrap().header.timestamp {
            return false;
        }
        if !self.validate_transactions(&block.transactions) {
            return false;
        }
        true
    }

    fn validate_transactions(&self, transactions: &Vec<Transaction>) -> bool {
        for transaction in transactions {
            if !self.validate_transaction(transaction) {
                return false;
            }
        }
        true
    }

    fn validate_transaction(&self, transaction: &Transaction) -> bool {
        // TO DO: implement transaction validation logic
        true
    }

    pub fn mine_block(&mut self) {
        let block = self.create_block();
        if self.validate_block(&block) {
            self.chain.push(block);
            self.pending_transactions.clear();
        } else {
            println!("Invalid block!");
        }
    }
}
