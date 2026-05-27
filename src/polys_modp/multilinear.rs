//! Dense multilinear polynomial over `SumcheckField`.

use crate::{polys_modp::eq::EqPolynomial, traits::mod_engine::SumcheckField};
use core::ops::Index;

/// Dense MLE stored as evaluations over `{0,1}^num_vars`. Length is a
/// power of two.
pub struct MultilinearPolynomial<F: SumcheckField> {
  pub(crate) Z: Vec<F>,
}

impl<F: SumcheckField> MultilinearPolynomial<F> {
  /// New from evaluation vector. Panics if length isn't a power of two.
  pub fn new(Z: Vec<F>) -> Self {
    assert!(
      Z.len().is_power_of_two(),
      "MultilinearPolynomial length must be a power of two"
    );
    Self { Z }
  }

  /// Number of stored evaluations (i.e. `2^num_vars`).
  pub fn len(&self) -> usize {
    self.Z.len()
  }

  /// Whether the polynomial has no evaluations.
  pub fn is_empty(&self) -> bool {
    self.Z.is_empty()
  }

  /// Number of free variables remaining (`log2(len)`).
  pub fn num_vars(&self) -> usize {
    self.Z.len().trailing_zeros() as usize
  }

  /// Take ownership of the underlying evaluation vector.
  pub fn into_vec(self) -> Vec<F> {
    self.Z
  }

  /// Bind the top (highest-index) variable to `r`, halving the
  /// evaluation table in place:
  ///
  ///   `Z'[i] = (1 - r) · Z[i] + r · Z[n + i]`  for `i ∈ [0, n)`,
  ///
  /// where `n = len() / 2`.
  pub fn bind_poly_var_top(&mut self, r: &F) {
    assert!(
      self.Z.len() >= 2,
      "cannot bind a variable of a 1-element polynomial"
    );
    let n = self.Z.len() / 2;
    let (left, right) = self.Z.split_at_mut(n);
    for i in 0..n {
      // left[i] <- left[i] + r * (right[i] - left[i])
      let delta = right[i] - left[i];
      left[i] += *r * delta;
    }
    self.Z.truncate(n);
  }

  /// Evaluate at `r ∈ F^num_vars`. Computes `sum_k eq(r, k) · Z[k]`.
  pub fn evaluate(&self, r: &[F]) -> F {
    assert_eq!(r.len(), self.num_vars());
    let chis = EqPolynomial::evals_from_points(r);
    chis.iter().zip(self.Z.iter()).map(|(c, z)| *c * *z).sum()
  }
}

impl<F: SumcheckField> Index<usize> for MultilinearPolynomial<F> {
  type Output = F;
  fn index(&self, i: usize) -> &F {
    &self.Z[i]
  }
}

#[cfg(test)]
mod tests {
  use super::*;
  use crate::provider::PallasHyraxEngine;
  use crate::traits::Engine;

  type F = <PallasHyraxEngine as Engine>::Scalar;

  #[test]
  fn bind_matches_evaluate_for_constant_dimension() {
    // For a 2-variable polynomial p(x_0, x_1), binding x_1 to r and
    // evaluating the result at x_0 = s should equal p(s, r).
    let z: Vec<F> = (0..4).map(|i| F::from((11 + i) as u64)).collect();
    let mut p = MultilinearPolynomial::new(z.clone());
    let r = F::from(7);
    p.bind_poly_var_top(&r);
    let s = F::from(13);
    let bound_eval = p.evaluate(&[s]);

    let full = MultilinearPolynomial::new(z);
    let full_eval = full.evaluate(&[s, r]);
    assert_eq!(bound_eval, full_eval);
  }

  #[test]
  fn evaluate_matches_dot_product() {
    let z: Vec<F> = (0..8).map(|i| F::from((i + 1) as u64)).collect();
    let p = MultilinearPolynomial::new(z.clone());
    let r: Vec<F> = (0..3).map(|i| F::from((5 + i) as u64)).collect();
    let chis = EqPolynomial::evals_from_points(&r);
    let expected: F = chis.iter().zip(z.iter()).map(|(c, v)| *c * *v).sum();
    assert_eq!(p.evaluate(&r), expected);
  }
}
