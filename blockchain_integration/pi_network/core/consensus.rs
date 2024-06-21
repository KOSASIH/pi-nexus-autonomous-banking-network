// Consensus algorithm implementation using a hybrid of PBFT and Delegated Proof of Stake (DPoS)
use crate::blockchain::{Block, Blockchain};
use crate::network::{Network, Node};
use crate::transaction::{Transaction, TransactionPool};

pub struct Consensus {
    blockchain: Blockchain,
    network: Network,
    transaction_pool: TransactionPool,
}

impl Consensus {
    pub fn new(blockchain: Blockchain, network: Network, transaction_pool: TransactionPool) -> Self {
        Consensus {
            blockchain,
            network,
            transaction_pool,
        }
    }

    pub fn propose_block(&mut self, node: &Node) -> Result<Block, String> {
        // PBFT-inspired proposal mechanism
        let mut block = Block::new(node);
        block.transactions = self.transaction_pool.get_transactions();
        self.network.broadcast_block_proposal(block.clone());
        Ok(block)
    }

    pub fn vote_on_block(&mut self, block: &Block) -> Result<(), String> {
        // DPoS-inspired voting mechanism
        let mut votes = 0;
        for node in self.network.nodes() {
            if node.vote_on_block(block) {
                votes += 1;
            }
        }
        if votes > self.network.nodes().len() / 2 {
            self.blockchain.add_block(block);
            Ok(())
        } else {
            Err("Block not accepted by the network".to_string())
        }
    }
}
