// node.rs

use std::collections::{HashMap, VecDeque};
use std::net::{TcpListener, TcpStream};
use std::sync::{Arc, Mutex};
use std::thread;

use crypto::{aes, hash, rsa};
use serde::{Deserialize, Serialize};

// Define the Node struct
pub struct Node {
    id: u64,
    public_key: rsa::PublicKey,
    private_key: rsa::PrivateKey,
    network_id: u64,
    peers: HashMap<u64, TcpStream>,
    blockchain: Vec<Block>,
    mempool: VecDeque<Transaction>,
    node_state: NodeState,
}

// Define the NodeState enum
enum NodeState {
    Idle,
    Syncing,
    Mining,
}

// Define the Block struct
#[derive(Serialize, Deserialize)]
pub struct Block {
    id: u64,
    transactions: Vec<Transaction>,
    previous_hash: [u8; 32],
    timestamp: u64,
}

// Define the Transaction struct
#[derive(Serialize, Deserialize)]
pub struct Transaction {
    id: u64,
    from: [u8; 20],
    to: [u8; 20],
    amount: u64,
}

impl Node {
    // Create a new node
    pub fn new(id: u64, public_key: rsa::PublicKey, private_key: rsa::PrivateKey, network_id: u64) -> Self {
        Node {
            id,
            public_key,
            private_key,
            network_id,
            peers: HashMap::new(),
            blockchain: Vec::new(),
            mempool: VecDeque::new(),
            node_state: NodeState::Idle,
        }
    }

    // Add a peer to the node
    pub fn add_peer(&mut self, peer_id: u64, stream: TcpStream) {
        self.peers.insert(peer_id, stream);
    }

    // Remove a peer from the node
    pub fn remove_peer(&mut self, peer_id: u64) {
        self.peers.remove(&peer_id);
    }

    // Send a message to a peer
    pub fn send_message(&self, peer_id: u64, message: &str) {
        if let Some(stream) = self.peers.get(&peer_id) {
            let encrypted_message = aes::encrypt(message, &self.public_key);
            stream.write_all(&encrypted_message).unwrap();
        }
    }

    // Receive a message from a peer
    pub fn receive_message(&mut self, message: &str) {
        let decrypted_message = aes::decrypt(message, &self.private_key);
        match decrypted_message {
            "block" => {
                // Handle block message
                let block: Block = serde_json::from_str(decrypted_message).unwrap();
                self.blockchain.push(block);
            }
            "transaction" => {
                // Handle transaction message
                let transaction: Transaction = serde_json::from_str(decrypted_message).unwrap();
                self.mempool.push_back(transaction);
            }
            _ => {
                // Handle unknown message
                println!("Unknown message: {}", decrypted_message);
            }
        }
    }

    // Start the node
    pub fn start(&mut self) {
        thread::spawn(move || {
            // Start listening for incoming connections
            let listener = TcpListener::bind("0.0.0.0:8080").unwrap();
            for stream in listener.incoming() {
                match stream {
                    Ok(stream) => {
                        // Handle incoming connection
                        self.handle_incoming_connection(stream);
                    }
                    Err(e) => {
                        println!("Error: {}", e);
                    }
                }
            }
        });
    }

    // Handle incoming connection
    fn handle_incoming_connection(&mut self, stream: TcpStream) {
        // Read the peer's ID and public key
        let mut buffer = [0; 1024];
        stream.read(&mut buffer).unwrap();
        let peer_id = u64::from_le_bytes(buffer[..8].try_into().unwrap());
        let peer_public_key = rsa::PublicKey::from_bytes(&buffer[8..]);

        // Add the peer to the node
        self.add_peer(peer_id, stream);

        // Send the node's ID and public key to the peer
        let message = format!("id:{}public_key:{}", self.id, self.public_key);
        self.send_message(peer_id, &message);
    }
}

fn main() {
    // Create a new node
    let node = Node::new(1, rsa::generate_keypair().0, rsa::generate_keypair().1, 1);

    // Start the node
    node.start();
}
