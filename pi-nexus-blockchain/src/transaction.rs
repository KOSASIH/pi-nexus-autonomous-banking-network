// transaction.rs (new)

use crate::utils::{hash, verify_signature};
use elliptic_curve::{KeyPair, PublicKey, Signature};

pub struct Transaction {
    pub sender: PublicKey,
    pub recipient: PublicKey,
    pub amount: u64,
    pub signature: Signature,
}

impl Transaction {
    pub fn new(sender: &KeyPair, recipient: &PublicKey, amount: u64) -> Self {
        let data = format!("{}{}{}", sender.public_key, recipient, amount);
        let signature = sender.sign(&data);
        Transaction {
            sender: sender.public_key.clone(),
            recipient: recipient.clone(),
            amount,
            signature,
        }
    }

    pub fn verify(&self) -> bool {
        let data = format!("{}{}{}", self.sender, self.recipient, self.amount);
        verify_signature(&self.sender, &data, &self.signature)
    }
}
