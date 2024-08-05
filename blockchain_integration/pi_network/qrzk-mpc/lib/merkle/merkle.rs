// Import necessary libraries and dependencies
extern crate sha2;
extern crate hex;

use sha2::{Sha256, Digest};
use hex::{FromHex, ToHex};

// Define the Merkle tree struct
pub struct MerkleTree {
    root: Vec<u8>,
    leaves: Vec<Vec<u8>>,
}

// Implement the Merkle tree struct
impl MerkleTree {
    // Initialize the Merkle tree instance
    pub fn new(leaves: Vec<Vec<u8>>) -> Self {
        let mut tree = MerkleTree { root: vec![], leaves };
        tree.build_tree();
        tree
    }

    // Build the Merkle tree
    fn build_tree(&mut self) {
        let mut nodes = self.leaves.clone();
        while nodes.len() > 1 {
            let mut new_nodes = vec![];
            for i in 0..nodes.len() / 2 {
                let left = nodes[i * 2].clone();
                let right = nodes[i * 2 + 1].clone();
                let node = self.hash_nodes(left, right);
                new_nodes.push(node);
            }
            if nodes.len() % 2 == 1 {
                new_nodes.push(nodes.last().unwrap().clone());
            }
            nodes = new_nodes;
        }
        self.root = nodes[0].clone();
    }

    // Hash two nodes together
    fn hash_nodes(&self, left: Vec<u8>, right: Vec<u8>) -> Vec<u8> {
        let mut hasher = Sha256::new();
        hasher.update(left);
        hasher.update(right);
        let result = hasher.finalize();
        result.to_vec()
    }

    // Get the Merkle root
    pub fn get_root(&self) -> Vec<u8> {
        self.root.clone()
    }

    // Get a Merkle proof
    pub fn get_proof(&self, index: usize) -> Vec<Vec<u8>> {
        let mut proof = vec![];
        let mut node = self.leaves[index].clone();
        let mut siblings = vec![];
        for i in 0..self.leaves.len().next_power_of_two().trailing_zeros() as usize {
            siblings.push(node.clone());
            node = self.hash_nodes(node, siblings.last().unwrap().clone());
            proof.push(siblings.clone());
            siblings.clear();
        }
        proof
    }

    // Verify a Merkle proof
    pub fn verify_proof(&self, proof: Vec<Vec<u8>>, index: usize, root: Vec<u8>) -> bool {
        let mut node = self.leaves[index].clone();
        for sibling in proof {
            node = self.hash_nodes(node, sibling);
        }
        node == root
    }
}

// Example usage
fn main() {
    let leaves = vec![
        vec![1, 2, 3],
        vec![4, 5, 6],
        vec![7, 8, 9],
    ];
    let merkle_tree = MerkleTree::new(leaves);

    let root = merkle_tree.get_root();
    println!("Merkle root: {:?}", root);

    let proof = merkle_tree.get_proof(1);
    println!("Merkle proof: {:?}", proof);

    let verified = merkle_tree.verify_proof(proof, 1, root);
    println!("Verified: {}", verified);
}
