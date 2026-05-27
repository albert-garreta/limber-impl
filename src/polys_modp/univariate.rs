//! Dense univariate polynomial and its compressed (omit-linear-term)
//! form, used for sumcheck round polynomials.

use crate::{
  errors::SpartanError,
  traits::{mod_engine::SumcheckField, transcript::TranscriptReprTrait},
};

/// Univariate dense polynomial stored in **big-endian** order:
/// `coeffs[0]` is the constant term, `coeffs[k]` is the `x^k` term.
pub struct UniPoly<F: SumcheckField> {
  pub coeffs: Vec<F>,
}

/// `UniPoly` with the linear coefficient (`coeffs[1]`) omitted.
/// Recoverable from the round's running claim via
/// `p(0) + p(1) = claim`.
pub struct CompressedUniPoly<F: SumcheckField> {
  pub coeffs_except_linear_term: Vec<F>,
}

impl<F: SumcheckField> UniPoly<F> {
  /// New from coefficient vector (big-endian).
  pub fn new(coeffs: Vec<F>) -> Self {
    Self { coeffs }
  }

  /// Construct from evaluations at consecutive points `x = 0, 1, …, n-1`.
  /// Supports degree-2 (3 evals) and degree-3 (4 evals) — those are the
  /// shapes sumcheck round polynomials take.
  pub fn from_evals(evals: &[F]) -> Result<Self, SpartanError> {
    let two_inv = F::from(2).invert().ok_or(SpartanError::InternalError {
      reason: "2 is not invertible in this field".to_string(),
    })?;
    match evals.len() {
      3 => {
        // p(x) = a x^2 + b x + c
        // p(0) = c
        // p(1) = a + b + c
        // p(2) = 4a + 2b + c
        // → a = (p(0) - 2 p(1) + p(2)) / 2
        // → b = p(1) - p(0) - a
        let c = evals[0];
        let two_p1 = evals[1] + evals[1];
        let a = (evals[0] - two_p1 + evals[2]) * two_inv;
        let b = evals[1] - evals[0] - a;
        Ok(Self {
          coeffs: vec![c, b, a],
        })
      }
      4 => {
        // p(x) = a x^3 + b x^2 + c x + d
        // Lagrange interpolation via finite differences:
        //   Δ_i = p(i+1) - p(i)
        //   ΔΔ_i = Δ_{i+1} - Δ_i  (constant up to 2a)
        //   ΔΔΔ = ΔΔ_1 - ΔΔ_0 = 6a
        // → a = ΔΔΔ / 6
        // → b = ΔΔ_0 / 2 - 3a
        // → c = Δ_0 - a - b
        // → d = p(0)
        let six_inv = F::from(6).invert().ok_or(SpartanError::InternalError {
          reason: "6 is not invertible in this field".to_string(),
        })?;
        let d = evals[0];
        let delta1 = evals[1] - evals[0];
        let delta2 = evals[2] - evals[1];
        let delta3 = evals[3] - evals[2];
        let dd1 = delta2 - delta1;
        let dd2 = delta3 - delta2;
        let ddd = dd2 - dd1;
        let a = ddd * six_inv;
        let three_a = a + a + a;
        let b = dd1 * two_inv - three_a;
        let c = delta1 - a - b;
        Ok(Self {
          coeffs: vec![d, c, b, a],
        })
      }
      n => Err(SpartanError::InternalError {
        reason: format!("UniPoly::from_evals supports degree 2 or 3 only, got {n} evals"),
      }),
    }
  }

  /// Polynomial degree.
  pub fn degree(&self) -> usize {
    self.coeffs.len().saturating_sub(1)
  }

  /// Evaluate at `x` via Horner's rule.
  pub fn evaluate(&self, x: &F) -> F {
    let mut acc = F::zero();
    for c in self.coeffs.iter().rev() {
      acc = acc * *x + *c;
    }
    acc
  }

  /// Drop the linear coefficient. It can be recovered from the running
  /// sumcheck claim via `p(0) + p(1) = claim`.
  pub fn compress(&self) -> CompressedUniPoly<F> {
    assert!(
      self.coeffs.len() >= 2,
      "compress requires at least a linear term to omit"
    );
    let mut omitted = Vec::with_capacity(self.coeffs.len() - 1);
    omitted.push(self.coeffs[0]);
    omitted.extend_from_slice(&self.coeffs[2..]);
    CompressedUniPoly {
      coeffs_except_linear_term: omitted,
    }
  }
}

impl<F: SumcheckField> CompressedUniPoly<F> {
  /// Recover the full polynomial given the previous round's claim
  /// `hint = p(0) + p(1)`.
  ///
  /// `p(1) = sum of all coeffs`, `p(0) = coeffs[0]`. So
  /// `coeffs[1] = hint - 2·coeffs[0] - (coeffs[2] + coeffs[3] + …)`.
  pub fn decompress(&self, hint: &F) -> UniPoly<F> {
    assert!(
      !self.coeffs_except_linear_term.is_empty(),
      "compressed polynomial must contain at least the constant term"
    );
    let c0 = self.coeffs_except_linear_term[0];
    let high_sum: F = self.coeffs_except_linear_term[1..].iter().copied().sum();
    let c1 = *hint - c0 - c0 - high_sum;

    let mut coeffs = Vec::with_capacity(self.coeffs_except_linear_term.len() + 1);
    coeffs.push(c0);
    coeffs.push(c1);
    coeffs.extend_from_slice(&self.coeffs_except_linear_term[1..]);
    UniPoly { coeffs }
  }

  /// Number of stored coefficients (= `degree + 1` of the original poly,
  /// minus one for the omitted linear term).
  pub fn degree(&self) -> usize {
    self.coeffs_except_linear_term.len()
  }
}

impl<F: SumcheckField> TranscriptReprTrait for UniPoly<F> {
  fn to_transcript_bytes(&self) -> Vec<u8> {
    // Absorb the compressed coefficients so the bytes match what the
    // verifier reconstructs. Each coefficient is serialized little-
    // endian via `SumcheckField::to_le_bytes`.
    self
      .compress()
      .coeffs_except_linear_term
      .iter()
      .flat_map(|c| c.to_le_bytes())
      .collect()
  }
}

#[cfg(test)]
mod tests {
  use super::*;
  use crate::provider::PallasHyraxEngine;
  use crate::traits::Engine;

  type F = <PallasHyraxEngine as Engine>::Scalar;

  #[test]
  fn from_evals_then_evaluate_roundtrips_degree_2() {
    // Pick arbitrary coefficients, evaluate at 0/1/2, reconstruct,
    // and check that the reconstructed poly evaluates the same.
    let coeffs = vec![F::from(7), F::from(11), F::from(13)]; // 13 x^2 + 11 x + 7
    let original = UniPoly::new(coeffs.clone());
    let evals: Vec<F> = (0..3).map(|i| original.evaluate(&F::from(i))).collect();
    let recovered = UniPoly::from_evals(&evals).unwrap();
    assert_eq!(recovered.coeffs, coeffs);
  }

  #[test]
  fn from_evals_then_evaluate_roundtrips_degree_3() {
    let coeffs = vec![F::from(2), F::from(3), F::from(5), F::from(7)];
    let original = UniPoly::new(coeffs.clone());
    let evals: Vec<F> = (0..4).map(|i| original.evaluate(&F::from(i))).collect();
    let recovered = UniPoly::from_evals(&evals).unwrap();
    assert_eq!(recovered.coeffs, coeffs);
  }

  #[test]
  fn compress_decompress_roundtrip() {
    let coeffs = vec![F::from(7), F::from(11), F::from(13), F::from(17)];
    let p = UniPoly::new(coeffs.clone());
    let compressed = p.compress();
    // claim = p(0) + p(1)
    let claim = p.evaluate(&F::zero()) + p.evaluate(&F::one());
    let recovered = compressed.decompress(&claim);
    assert_eq!(recovered.coeffs, coeffs);
  }
}
