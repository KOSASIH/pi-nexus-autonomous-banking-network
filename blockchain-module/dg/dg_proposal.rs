// Import necessary libraries
use std::collections::HashMap;
use std::hash::{Hash, Hasher};
use std::io::{Read, Write};
use std::ops::{Add, Mul, Sub};
use std::vec::Vec;

// Define the Proposal struct
pub struct Proposal {
    // Proposal ID
    id: u32,
    // Proposal title
    title: String,
    // Proposal description
    description: String,
    // Proposal status
    status: ProposalStatus,
}

// Implement the Proposal struct
impl Proposal {
    // Create a new Proposal
    pub fn new(id: u32, title: String, description: String) -> Self {
        Proposal {
            id,
            title,
            description,
            status: ProposalStatus::Pending,
        }
    }
}

// Define the ProposalStatus enum
pub enum ProposalStatus {
    Pending,
    Voting,
    Approved,
    Rejected,
}

// Export the Proposal and ProposalStatus
pub use Proposal;
pub use ProposalStatus;
