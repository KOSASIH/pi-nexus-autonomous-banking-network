// zk-snarks.rs

use crate::curve::{Curve, Point};
use crate::field::{Field, Fp};
use crate::hash::{Hash, Sha256};
use crate::poly::{Poly, PolyCommit};
use crate::proof::{Proof, ProofSystem};
use crate::utils::{bytes_to_fp, fp_to_bytes};

// zk-SNARKs protocol implementation
pub struct ZkSnarks {
    curve: Curve,
    field: Field,
    hash: Hash,
    poly_commit: PolyCommit,
}

impl ZkSnarks {
    pub fn new(curve: Curve, field: Field, hash: Hash, poly_commit: PolyCommit) -> Self {
        ZkSnarks {
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

// zk-SNARKs proof system implementation
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

// zk-SNARKs polynomial commitment implementation
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

// zk-SNARKs polynomial implementation
struct Poly {
    coeffs: Vec<Fp>,
}

impl Poly {
    fn new(statement: &[u8], witness: &[u8]) -> Self {
        // Compute the polynomial coefficients
        let coeffs = Self::compute_coeffs(statement, witness);

        Poly { coeffs }
    }

    fn compute_coeffs(statement: &[u8], witness: &[u8]) -> Vec<Fp> {
        // Compute the polynomial coefficients using the witness
        let mut coeffs = Vec::new();

        for i in 0..statement.len() {
            let coeff = Fp::from_bytes(&statement[i..i+32]);
            coeffs.push(coeff);
        }

        coeffs
    }

    fn eval(&self, public_input: &[u8]) -> Vec<u8> {
        // Evaluate the polynomial at the public input
        let mut eval = Vec::new();

        for i in 0..public_input.len() {
            let x = Fp::from_bytes(&public_input[i..i+32]);
            let y = self.eval_at(x);
            eval.extend_from_slice(&y.to_bytes());
        }

        eval
    }

    fn eval_at(&self, x: Fp) -> Fp {
        // Evaluate the polynomial at a single point
        let mut result = Fp::zero();

        for (i, coeff) in self.coeffs.iter().enumerate() {
            let term = coeff * x.pow(i as u32);
            result += term;
        }

        result
    }
}
   
