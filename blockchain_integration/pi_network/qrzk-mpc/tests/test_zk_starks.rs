// tests/test_zk_starks.rs: Test suite for zk-STARKs implementation

use zk_starks::{Curve, Poly, PolyCommit, Proof, ProofSystem};
use rand::OsRng;
use hex;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_zk_starks_proof_generation() {
        // Generate a random curve
        let curve = Curve::from_str("G1,G2,N").unwrap();

        // Generate a random polynomial
        let mut poly = Poly::new();
        poly.set_coeff(0, Scalar::random(&mut OsRng));
        poly.set_coeff(1, Scalar::random(&mut OsRng));
        poly.set_coeff(2, Scalar::random(&mut OsRng));

                // Generate a random input
        let input = b"Hello, zk-STARKs!";

        // Generate a random hash
        let hash = Scalar::random(&mut OsRng);

        // Create a proof system
        let proof_system = ProofSystem::new(curve.clone());

        // Generate a proof
        let mut transcript = Transcript::new();
        let proof = proof_system.prove(input, &hash, &mut transcript).unwrap();

        // Verify the proof
        let verified = proof_system.verify(&proof, input, &hash, &mut transcript).unwrap();
        assert!(verified);
    }

    #[test]
    fn test_zk_starks_proof_serialization() {
        // Generate a random curve
        let curve = Curve::from_str("G1,G2,N").unwrap();

        // Generate a random polynomial
        let mut poly = Poly::new();
        poly.set_coeff(0, Scalar::random(&mut OsRng));
        poly.set_coeff(1, Scalar::random(&mut OsRng));
        poly.set_coeff(2, Scalar::random(&mut OsRng));

        // Generate a random input
        let input = b"Hello, zk-STARKs!";

        // Generate a random hash
        let hash = Scalar::random(&mut OsRng);

        // Create a proof system
        let proof_system = ProofSystem::new(curve.clone());

        // Generate a proof
        let mut transcript = Transcript::new();
        let proof = proof_system.prove(input, &hash, &mut transcript).unwrap();

        // Serialize the proof
        let proof_bytes = proof.to_bytes();

        // Deserialize the proof
        let deserialized_proof = Proof::from_bytes(&proof_bytes).unwrap();

        // Verify the deserialized proof
        let verified = proof_system.verify(&deserialized_proof, input, &hash, &mut transcript).unwrap();
        assert!(verified);
    }

    #[test]
    fn test_zk_starks_proof_invalid_input() {
        // Generate a random curve
        let curve = Curve::from_str("G1,G2,N").unwrap();

        // Generate a random polynomial
        let mut poly = Poly::new();
        poly.set_coeff(0, Scalar::random(&mut OsRng));
        poly.set_coeff(1, Scalar::random(&mut OsRng));
        poly.set_coeff(2, Scalar::random(&mut OsRng));

        // Generate a random input
        let input = b"Hello, zk-STARKs!";

        // Generate a random hash
        let hash = Scalar::random(&mut OsRng);

        // Create a proof system
        let proof_system = ProofSystem::new(curve.clone());

        // Generate a proof
        let mut transcript = Transcript::new();
        let proof = proof_system.prove(input, &hash, &mut transcript).unwrap();

        // Verify the proof with an invalid input
        let invalid_input = b"Goodbye, zk-STARKs!";
        let verified = proof_system.verify(&proof, invalid_input, &hash, &mut transcript).unwrap();
        assert!(!verified);
    }
}
