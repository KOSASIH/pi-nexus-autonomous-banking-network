// utils.rs (new)

use elliptic_curve::{KeyPair, PublicKey, Signature};

pub fn hash(data: &str) -> String {
    // TO DO: implement hash function
    data.to_string()
}

pub fn verify_signature(public_key: &PublicKey, data: &str, signature: &Signature) -> bool {
    // TO DO: implement signature verification
    true
}
