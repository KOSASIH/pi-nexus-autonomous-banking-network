// block.rs (updated)

// ...

impl BlockHeader {
    // ...

    pub fn mine(&mut self, difficulty_target: u64) {
        let mut nonce = 0;
        loop {
            self.nonce = nonce;
            let hash = self.hash();
            if hash.starts_with(&format!("{:0>64}", difficulty_target)) {
                break;
            }
            nonce += 1;
        }
    }
}
