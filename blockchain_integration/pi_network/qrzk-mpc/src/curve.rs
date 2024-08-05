// curve.rs: Curve implementation for the QRZK-MPC system

use curve25519_dalek::{ristretto::CompressedRistretto, scalar::Scalar};

pub struct Curve {
    pub g: CompressedRistretto,
    pub h: CompressedRistretto,
    pub n: Scalar,
}

impl Curve {
    pub fn from_str(s: &str) -> Result<Self, Box<dyn std::error::Error>> {
        // Parse the curve parameters from the string
        let params: Vec<&str> = s.split(',').collect();
        let g = CompressedRistretto::from_bytes(hex::decode(params[0])?.as_slice())?;
        let h = CompressedRistretto::from_bytes(hex::decode(params[1])?.as_slice())?;
        let n = Scalar::from_bytes(hex::decode(params[2])?.as_slice())?;

        Ok(Curve { g, h, n })
    }
}

pub struct Point {
    pub x: CompressedRistretto,
    pub y: CompressedRistretto,
}

impl Point {
    pub fn new(x: CompressedRistretto, y: CompressedRistretto) -> Self {
        Point { x, y }
    }

    pub fn add(&self, other: &Point, curve: &Curve) -> Point {
        // Add two points on the curve
        let x = self.x + other.x;
        let y = self.y + other.y;
        let z = curve.g * curve.n;

        Point { x, y }
    }

    pub fn scalar_mul(&self, scalar: &Scalar, curve: &Curve) -> Point {
        // Multiply a point by a scalar
        let x = self.x * scalar;
        let y = self.y * scalar;

        Point { x, y }
    }
}
