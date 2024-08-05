// zk-starks.rs

use crate::curve::{Curve, Point};
use crate::field::{Field, Fp};
use crate::hash::{Hash, Sha256};
use crate::poly::{Poly, PolyCommit};
use crate::proof::{Proof, ProofSystem};
use crate::utils::{bytes_to_fp, fp_to_bytes};

// zk-STARKs protocol implementation
pub struct ZkStarks {
    curve: Curve,
    field: Field,
    hash: Hash,
    poly_commit: PolyCommit,
}

impl ZkStarks {
    pub fn new(curve: Curve, field: Field, hash: Hash, poly_commit: PolyCommit) -> Self {
        ZkStarks {
            curve,
            field,
            hash,
            poly_commit,
        }
    }

    pub fn generate_proof(
        &self,
        statement: &[u8],
        witness: &[u8],
        public_input: &[u8],
    ) -> Result<Proof, String> {
        // 1. Compute the polynomial commitment
        let poly_commitment = self.poly_commit.commit(statement, witness);

        // 2. Compute the polynomial evaluation
        let poly_eval = self.poly_commit.eval(poly_commitment, public_input);

        // 3. Compute the hash of the polynomial evaluation
        let hash = self.hash.hash(&poly_eval);

        // 4. Compute the proof
        let proof = self.proof_system.prove(statement, witness, public_input, hash);

        Ok(proof)
    }

    pub fn verify_proof(
        &self,
        proof: &Proof,
        statement: &[u8],
        public_input: &[u8],
    ) -> Result<bool, String> {
        // 1. Compute the polynomial commitment
        let poly_commitment = self.poly_commit.commit(statement, &[]);

        // 2. Compute the polynomial evaluation
        let poly_eval = self.poly_commit.eval(poly_commitment, public_input);

        // 3. Compute the hash of the polynomial evaluation
        let hash = self.hash.hash(&poly_eval);

        // 4. Verify the proof
        let result = self.proof_system.verify(proof, statement, public_input, hash);

        Ok(result)
    }
}

// zk-STARKs proof system implementation
struct ProofSystem {
    curve: Curve,
    field: Field,
}

impl ProofSystem {
    fn prove(
        &self,
        statement: &[u8],
        witness: &[u8],
        public_input: &[u8],
        hash: &[u8],
    ) -> Proof {
        // 1. Compute the polynomial commitment
        let poly_commitment = PolyCommit::commit(statement, witness);

        // 2. Compute the polynomial evaluation
        let poly_eval = PolyCommit::eval(poly_commitment, public_input);

        // 3. Compute the proof
        let proof = self.generate_proof(poly_eval, hash);

        proof
    }

    fn verify(
        &self,
        proof: &Proof,
        statement: &[u8],
        public_input: &[u8],
        hash: &[u8],
    ) -> bool {
        // 1. Compute the polynomial commitment
        let poly_commitment = PolyCommit::commit(statement, &[]);

        // 2. Compute the polynomial evaluation
        let poly_eval = PolyCommit::eval(poly_commitment, public_input);

        // 3. Verify the proof
        self.verify_proof(proof, poly_eval, hash)
    }

    fn generate_proof(&self, poly_eval: &[u8], hash: &[u8]) -> Proof {
        // Generate a random point on the curve
        let point = self.curve.random_point();

        // Compute the proof
        let proof = Proof {
            point,
            scalar: self.field.random_scalar(),
        };

        proof
    }

    fn verify_proof(&self, proof: &Proof, poly_eval: &[u8], hash: &[u8]) -> bool {
        // Compute the point on the curve
        let point = self.curve.add(proof.point, proof.scalar);

        // Compute the hash of the point
        let hash_point = self.hash.hash(&point.to_bytes());

        // Check if the hash matches
        hash_point == hash
    }
}

// zk-STARKs polynomial commitment implementation
struct PolyCommit {
    curve: Curve,
    field: Field,
}

impl PolyCommit {
    fn commit(statement: &[u8], witness: &[u8]) -> Poly {
        // Compute the polynomial commitment
        let poly = Poly::new(statement, witness);

        poly
    }

    fn eval(poly: &Poly, public_input: &[u8]) -> Vec<u8> {
        // Compute the polynomial evaluation
        let eval = poly.eval(public_input);

        eval
    }
}
