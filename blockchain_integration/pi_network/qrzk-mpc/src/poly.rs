// poly.rs: Polynomial implementation for the QRZK-MPC system

use std::collections::HashMap;

pub struct Poly {
    coeffs: HashMap<usize, Scalar>,
}

impl Poly {
    pub fn new() -> Self {
        Poly { coeffs: HashMap::new() }
    }

    pub fn set_coeff(&mut self, degree: usize, coeff: Scalar) {
        self.coeffs.insert(degree, coeff);
    }

    pub fn eval(&self, x: &Scalar) -> Scalar {
        // Evaluate the polynomial at a point
        let mut result = Scalar::zero();

        for (degree, coeff) in &self.coeffs {
            let term = coeff * x.pow(*degree as u32);
            result += term;
        }

        result
    }
}

pub struct PolyCommit {
    curve: Curve,
}

impl PolyCommit {
    pub fn new(curve: Curve) -> Self {
        PolyCommit { curve }
    }

    pub fn commit(&self, poly: &Poly) -> Point {
        // Commit to a polynomial
        let mut commitment = Point::new(self.curve.g, self.curve.h);

        for (degree, coeff) in &poly.coeffs {
            let point = self.curve.g * coeff;
            commitment = commitment.add(&point, &self.curve);
        }

        commitment
    }

    pub fn eval(&self, commitment: &Point, poly: &Poly) -> Scalar {
        // Evaluate the polynomial commitment
        let mut result = Scalar::zero();

        for (degree, coeff) in &poly.coeffs {
            let term = coeff * commitment.x.pow(*degree as u32);
            result += term;
        }

        result
    }
}
