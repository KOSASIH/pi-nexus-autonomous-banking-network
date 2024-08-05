// lib.rs: Library implementation for the QRZK-MPC system

use std::collections::HashMap;
use std::error::Error;
use std::fmt;

use curve25519_dalek::{ristretto::CompressedRistretto, scalar::Scalar};
use merlin::Transcript;
use rand_core::{OsRng, RngCore};
use sha2::{Digest, Sha256};

mod curve;
mod poly;
mod proof;

pub use curve::{Curve, Point};
pub use poly::{Poly, PolyCommit};
pub use proof::{Proof, ProofSystem};

// Configuration for the QRZK-MPC system
#[derive(Debug, Clone)]
pub struct Config {
    pub curve: Curve,
    pub hash: Sha256,
    pub poly_commit: PolyCommit,
    pub proof_system: ProofSystem,
}

impl Config {
    pub fn from_file(path: &str) -> Result<Self, Box<dyn Error>> {
        // Load configuration from file
        let config_data = fs::read_to_string(path)?;
        let config: HashMap<String, String> = serde_json::from_str(&config_data)?;

        // Create a new configuration instance
        let curve = Curve::from_str(&config["curve"])?;
        let hash = Sha256::new();
        let poly_commit = PolyCommit::new(curve.clone());
        let proof_system = ProofSystem::new(curve.clone());

        Ok(Config {
            curve,
            hash,
            poly_commit,
            proof_system,
        })
    }
}

// QRZK-MPC system implementation
pub struct QRZKMPC {
    config: Config,
    transcript: Transcript,
}

impl QRZKMPC {
    pub fn new(config: Config) -> Self {
        let transcript = Transcript::new(b"QRZK-MPC");

        QRZKMPC { config, transcript }
    }

    pub fn generate_proof(&mut self, input: Vec<u8>) -> Result<Proof, Box<dyn Error>> {
        // Compute the polynomial commitment
        let poly_commitment = self.config.poly_commit.commit(&input);

        // Compute the polynomial evaluation
        let poly_eval = self.config.poly_commit.eval(&poly_commitment, &input);

        // Compute the hash of the polynomial evaluation
        let hash = self.config.hash.hash(&poly_eval);

        // Generate a proof using the proof system
        let proof = self.config.proof_system.prove(&input, &hash, &mut self.transcript)?;

        Ok(proof)
    }
}

// Error type for the QRZK-MPC system
#[derive(Debug, Clone)]
pub enum QRZKMPCError {
    ConfigError(String),
    ProofError(String),
}

impl fmt::Display for QRZKMPCError {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self {
            QRZKMPCError::ConfigError(msg) => write!(f, "Config error: {}", msg),
            QRZKMPCError::ProofError(msg) => write!(f, "Proof error: {}", msg),
        }
    }
}

impl Error for QRZKMPCError {}
