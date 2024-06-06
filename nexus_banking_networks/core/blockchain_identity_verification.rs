use crate::zk_proof::{ZKProof, generate_proof};
use crate::blockchain::{Blockchain, Transaction};

// Define the blockchain-based identity verification system
struct BlockchainIdentityVerification {
    blockchain: Blockchain,
    zk_proof: ZKProof,
}

impl BlockchainIdentityVerification {
    fn new() -> Self {
        let blockchain = Blockchain::new();
        let zk_proof = ZKProof::new();
        BlockchainIdentityVerification { blockchain, zk_proof }
    }

    fn register_identity(&mut self, identity: String) {
        let transaction = Transaction::new(identity);
        self.blockchain.add_transaction(transaction);
    }

    fn verify_identity(&self, identity: String) -> bool {
        let proof = generate_proof(self.zk_proof, identity);
        self.blockchain.verify_transaction(proof)
    }
}

# Example usage
let mut biv = BlockchainIdentityVerification::new();
biv.register_identity("Alice".to_string());
assert!(biv.verify_identity("Alice".to_string()));
