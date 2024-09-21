// Import necessary libraries
use std::collections::HashMap;
use std::hash::{Hash, Hasher};
use std::io::{Read, Write};
use std::ops::{Add, Mul, Sub};
use std::vec::Vec;

// Import the DG voting and proposal modules
use crate::dg_voting::{VotingSystem, Vote};
use crate::dg_proposal::{Proposal, ProposalStatus};

// Define the Decentralized Governance (DG) module
pub struct DG {
    // Voting system
    voting_system: VotingSystem,
    // Proposal management system
    proposal_system: Vec<Proposal>,
}

// Implement the DG module
impl DG {
    // Create a new DG module
    pub fn new(voting_system: VotingSystem) -> Self {
        DG {
            voting_system,
            proposal_system: Vec::new(),
        }
    }

    // Submit a new proposal
    pub fn submit_proposal(&mut self, proposal: Proposal) {
        self.proposal_system.push(proposal);
    }

    // Vote on a proposal
    pub fn vote(&mut self, proposal_id: u32, vote: Vote) {
        // Get the proposal
        let proposal = self.proposal_system.get_mut(proposal_id as usize).unwrap();

        // Update the proposal status
        proposal.status = ProposalStatus::Voting;

        // Add the vote to the voting system
        self.voting_system.add_vote(proposal_id, vote);
    }

    // Get the current proposal status
    pub fn get_proposal_status(&self, proposal_id: u32) -> ProposalStatus {
        self.proposal_system.get(proposal_id as usize).unwrap().status
    }
}

// Export the DG module
pub use DG;
