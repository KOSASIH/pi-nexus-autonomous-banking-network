// Import necessary libraries
use std::collections::HashMap;
use std::hash::{Hash, Hasher};
use std::io::{Read, Write};
use std::ops::{Add, Mul, Sub};
use std::vec::Vec;

// Import the QRC module
use crate::qrc::{NTRU, McEliece};

// Define the test module
#[cfg(test)]
mod tests {
    // Test the NTRU algorithm
    #[test]
    fn test_ntru() {
        // Generate a new NTRU key pair
        let (ntru, _) = NTRU::new(256, 2048, 256);

        // Encrypt a message using the public key
        let message = vec![1, 2, 3, 4, 5];
        let ciphertext = ntru.encrypt(&message);

        // Decrypt the ciphertext using the private key
        let plaintext = ntru.decrypt(&ciphertext);

        // Verify that the decrypted plaintext matches the original message
        assert_eq!(plaintext, message);
    }

    // Test the McEliece algorithm
    #[test]
    fn test_mceliece() {
        // Generate a new McEliece key pair
        let (mceliece, _) = McEliece::new(256, 128, 32);

        // Encrypt a message using the public key
        let message = vec![1, 2, 3, 4, 5];
        let ciphertext = mceliece.encrypt(&message);

        // Decrypt the ciphertext using the private key
        let plaintext = mceliece.decrypt(&ciphertext);

        // Verify that the decrypted plaintext matches the original message
        assert_eq!(plaintext, message);
    }
}
