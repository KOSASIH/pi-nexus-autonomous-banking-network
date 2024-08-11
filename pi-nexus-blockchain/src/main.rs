// main.rs

use std::thread;
use std::time::Duration;

use crate::blockchain::{Blockchain, BlockchainNode};
use crate::transaction::{Transaction, TransactionPool};

fn main() {
    let node_id = "node-1".to_string();
    let blockchain_node = BlockchainNode::new(node_id);

    // Create some transactions
    let transaction1 = Transaction::new("Alice".to_string(), "Bob".to_string(), 10);
    let transaction2 = Transaction::new("Bob".to_string(), "Charlie".to_string(), 20);
    let transaction3 = Transaction::new("Charlie".to_string(), "Alice".to_string(), 30);

    // Add transactions to the transaction pool
    blockchain_node.add_transaction(transaction1);
    blockchain_node.add_transaction(transaction2);
    blockchain_node.add_transaction(transaction3);

    // Mine a block
    blockchain_node.mine_block();

    // Print the blockchain
    println!("Blockchain:");
    for block in blockchain_node.get_chain() {
        println!("Block {} - {}", block.header.index, block.hash());
    }

    // Start a new thread to mine blocks every 10 seconds
    thread::spawn(move || {
        loop {
            thread::sleep(Duration::from_secs(10));
            blockchain_node.mine_block();
        }
    });
}
