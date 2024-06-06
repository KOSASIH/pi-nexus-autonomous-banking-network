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

    fn evaluate(&self, cipher_text: CipherText, function: &str) -> CipherText {
        evaluate(self.public_key, cipher_text, function)
    }
}

# Example usage
let he = HomomorphicEncryption::new();
let plain_text = PlainText::from("Hello, World!");
let cipher_text = he.encrypt(plain_text);
let evaluated_cipher_text = he.evaluate(cipher_text, "uppercase");
let decrypted_text = he.decrypt(evaluated_cipher_text);
println!("Decrypted text: {}", decrypted_text);
