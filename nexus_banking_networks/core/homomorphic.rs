use crate::homomorphic::{CipherText, PlainText};

// Define the homomorphic encryption scheme
struct HomomorphicEncryption {
    public_key: PublicKey,
    private_key: PrivateKey,
}

impl HomomorphicEncryption {
    fn new() -> Self {
        let (public_key, private_key) = generate_keys();
        HomomorphicEncryption { public_key, private_key }
    }

    fn encrypt(&self, plain_text: PlainText) -> CipherText {
        encrypt(self.public_key, plain_text)
    }

    fn decrypt(&self, cipher_text: CipherText) -> PlainText {
        decrypt(self.private_key, cipher_text)
    }
}

// Example usage
fn main() {
    let he = HomomorphicEncryption::new();
    let plain_text = PlainText::from("Hello, World!");
    let cipher_text = he.encrypt(plain_text);
    let decrypted_text = he.decrypt(cipher_text);
    println!("Decrypted text: {}", decrypted_text);
}
