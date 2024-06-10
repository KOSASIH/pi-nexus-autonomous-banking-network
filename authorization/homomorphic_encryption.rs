// homomorphic_encryption.rs
use rust_homomorphic_encryption::{HomomorphicEncryption, PublicKey, Ciphertext};

struct SecureDataSharing {
    he: HomomorphicEncryption,
    public_key: PublicKey,
}

impl SecureDataSharing {
    fn new() -> Self {
        let he = HomomorphicEncryption::new();
        let public_key = he.generate_public_key();
        SecureDataSharing { he, public_key }
    }

    fn encrypt(&self, plaintext: &str) -> Ciphertext {
        self.he.encrypt(plaintext, &self.public_key)
    }

    fn decrypt(&self, ciphertext: &Ciphertext) -> String {
        self.he.decrypt(ciphertext, &self.public_key)
    }

    fn evaluate(&self, ciphertext1: &Ciphertext, ciphertext2: &Ciphertext) -> Ciphertext {
        self.he.evaluate(ciphertext1, ciphertext2, &self.public_key)
    }
}
