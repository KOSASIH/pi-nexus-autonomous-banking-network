// Import necessary libraries
use std::collections::HashMap;
use std::hash::{Hash, Hasher};
use std::io::{Read, Write};
use std::ops::{Add, Mul, Sub};
use std::vec::Vec;

// Define the VotingSystem struct
pub struct VotingSystem {
    // Voting data
    votes: HashMap<u32, Vec<Vote>>,
}

// Implement the VotingSystem struct
impl VotingSystem {
    // Create a new VotingSystem
    pub fn new() -> Self {
        VotingSystem {
            votes: HashMap::new(),
        }
    }

    // Add a vote to the voting system
    pub fn add_vote(&mut self, proposal_id: u32, vote: Vote) {
        // Get the vote vector for the proposal
        let vote_vector = self.votes.entry(proposal_id).or_insert(Vec::new());

        // Add the vote to the vector
        vote_vector.push(vote);
    }

    // Get the voting results for a proposal
    pub fn get_voting_results(&self, proposal_id: u32) -> Vec<Vote> {
        self.votes.get(&proposal_id).unwrap().clone()
    }
}

// Define the Vote enum
pub enum Vote {
    Yes,
    No,
    Abstain,
}

// Export the VotingSystem and Vote
pub use VotingSystem;
pub use Vote;
