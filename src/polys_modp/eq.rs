//! Equality polynomial `eq(r, x) = ∏_i (r_i x_i + (1-r_i)(1-x_i))`.

use crate::traits::mod_engine::SumcheckField;

/// Equality polynomial parameterized by a vector `r`.
pub struct EqPolynomial<F: SumcheckField> {
  pub r: Vec<F>,
}

impl<F: SumcheckField> EqPolynomial<F> {
  /// Construct an `EqPolynomial` with the given parameter vector.
  pub fn new(r: Vec<F>) -> Self {
    Self { r }
  }

  /// Evaluate `eq(self.r, rx)` for a single point `rx`.
  pub fn evaluate(&self, rx: &[F]) -> F {
    assert_eq!(self.r.len(), rx.len());
    let one = F::one();
    rx.iter()
      .zip(self.r.iter())
      .map(|(rxi, ri)| *rxi * *ri + (one - *rxi) * (one - *ri))
      .product()
  }

  /// Build the full eq-table: `evals[k] = eq(r, k)` for every
  /// `k ∈ {0,1}^{|r|}`, returned as a length-`2^|r|` vector.
  ///
  /// **Bit-ordering convention** (matches `src/polys/eq.rs`): `r` is
  /// consumed in **reverse** so that the resulting table aligns with
  /// the top-down binding order used by `MultilinearPolynomial::
  /// bind_poly_var_top`. After binding the top variable to challenge
  /// `r_0` (round 0), then `r_1` (round 1), …, the final value
  /// `evals[0]` of the bound table equals `eq(r, [r_0, r_1, …, r_{n-1}])`
  /// — i.e. the natural `EqPolynomial::new(r).evaluate(&challenges)`
  /// without reversal.
  pub fn evals_from_points(r: &[F]) -> Vec<F> {
    let one = F::one();
    let mut evals = vec![F::zero(); 1usize << r.len()];
    evals[0] = one;
    let mut size = 1usize;
    for &r_i in r.iter().rev() {
      let one_minus_r = one - r_i;
      for i in 0..size {
        let v = evals[i];
        evals[i] = v * one_minus_r;
        evals[size + i] = v * r_i;
      }
      size *= 2;
    }
    evals
  }
}

#[cfg(test)]
mod tests {
  use super::*;
  use crate::provider::PallasHyraxEngine;
  use crate::traits::Engine;

  type F = <PallasHyraxEngine as Engine>::Scalar;

  #[test]
  fn eq_table_consistent_with_evaluate() {
    // Convention (matches `src/polys/eq.rs`): `evals_from_points` consumes
    // `r` in reverse, so `table[k]`'s bit `i` corresponds to `r[len-1-i]`.
    // The matching `x` for direct evaluation reverses the bit ordering.
    let r: Vec<F> = (0..4).map(|i| F::from((7 + i) as u64)).collect();
    let table = EqPolynomial::evals_from_points(&r);
    let eq = EqPolynomial::new(r.clone());
    let n = r.len();
    for (k, &table_val) in table.iter().enumerate() {
      let x: Vec<F> = (0..n)
        .map(|i| {
          if (k >> (n - 1 - i)) & 1 == 1 {
            F::one()
          } else {
            F::zero()
          }
        })
        .collect();
      assert_eq!(eq.evaluate(&x), table_val);
    }
  }

  #[test]
  fn eq_table_sums_to_one() {
    // sum_k eq(r, k) = 1 (probability distribution property)
    let r: Vec<F> = (0..5).map(|i| F::from((3 * i + 1) as u64)).collect();
    let table = EqPolynomial::evals_from_points(&r);
    let sum: F = table.iter().copied().sum();
    assert_eq!(sum, F::one());
  }
}
