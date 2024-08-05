// proof.rs: Proof implementation for the QRZK-MPC system

use std::collections::HashMap;

pub struct Proof {
    a: Point,
    b: Point,
    c: Point,
}

impl Proof {
    pub fn new(a: Point, b: Point, c: Point) -> Self {
        Proof { a, b, c }
    }

    pub fn to_bytes(&self) -> Vec<u8> {
        // Serialize the proof to bytes
        let mut bytes = Vec::new();

        bytes.extend_from_slice(&self.a.x.to_bytes());
        bytes.extend_from_slice(&self.a.y.to_bytes());
        bytes.extend_from_slice(&self.b.x.to_bytes());
        bytes.extend_from_slice(&self.b.y.to_bytes());
        bytes.extend_from_slice(&self.c.x.to_bytes());
        bytes.extend_from_slice(&self.c.y.to_bytes());

        bytes
    }
}

pub struct ProofSystem {
    curve: Curve,
}

impl ProofSystem {
    pub fn new(curve: Curve) -> Self {
        ProofSystem { curve }
    }

        pub fn prove(&self, input: &[u8], hash: &Scalar, transcript: &mut Transcript) -> Result<Proof, Box<dyn std::error::Error>> {
        // Generate a random scalar
        let r = Scalar::random(&mut OsRng);

        // Compute the commitment to the input
        let commitment = self.curve.g * r;

        // Compute the challenge
        let challenge = transcript.challenge(b"challenge", &commitment, input, hash);

        // Compute the response
        let response = r + challenge * self.curve.n;

        // Compute the proof
        let a = self.curve.g * response;
        let b = self.curve.h * response;
        let c = commitment + challenge * self.curve.g;

        Ok(Proof { a, b, c })
    }

    pub fn verify(&self, proof: &Proof, input: &[u8], hash: &Scalar, transcript: &mut Transcript) -> Result<bool, Box<dyn std::error::Error>> {
        // Compute the challenge
        let challenge = transcript.challenge(b"challenge", &proof.a, input, hash);

        // Verify the proof
        let lhs = proof.a + challenge * self.curve.g;
        let rhs = proof.c + challenge * self.curve.h;

        Ok(lhs == rhs)
    }
}
