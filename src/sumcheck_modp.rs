// TODO Phase 2 step 5+: remove `allow(dead_code)` once a Phase-2
// IntModSpartan driver starts consuming this module.
#![allow(dead_code)]

//! Parallel sumcheck protocol bound on `SumcheckEngine`, used by Phase-2
//! IntMod-Spartan. Lives alongside `src/sumcheck.rs` rather than
//! replacing it — see [[project-phase2-parallel-sumcheck]] in memory
//! for the rationale.
//!
//! Deliberately minimal: no BDDT round-0 optimization, no Gruen
//! eq-factoring, no delayed-reduction Montgomery accumulators. Just the
//! protocol math, easy to read and verify. Optimizations can be added
//! later if benchmarks demand.

use crate::{
  errors::SpartanError,
  polys_modp::{
    eq::EqPolynomial,
    multilinear::MultilinearPolynomial,
    univariate::{CompressedUniPoly, UniPoly},
  },
  traits::{
    mod_engine::{SumcheckEngine, SumcheckField},
    transcript::TranscriptEngineTrait,
  },
};
use serde::{Deserialize, Serialize};

/// A sumcheck proof — the per-round compressed univariate polynomials.
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(bound = "")]
pub struct SumcheckProof<E: SumcheckEngine> {
  pub(crate) compressed_polys: Vec<CompressedUniPoly<E::Scalar>>,
}

impl<E: SumcheckEngine> SumcheckProof<E> {
  /// Construct from per-round compressed polys (used by `prove_*`).
  fn new(compressed_polys: Vec<CompressedUniPoly<E::Scalar>>) -> Self {
    Self { compressed_polys }
  }

  /// Verify the proof. Performs intra-round consistency
  /// (`p_i(0) + p_i(1) == running_claim`) and degree checks; returns
  /// `(final_claim, challenges)`. The caller is responsible for
  /// re-checking `final_claim` against the integrand's structure
  /// (e.g. for our outer SC, `eq(tau, r_x) * (v_a v_b - v_c - v_m v_q)`).
  pub fn verify(
    &self,
    initial_claim: E::Scalar,
    num_rounds: usize,
    degree_bound: usize,
    transcript: &mut E::TE,
  ) -> Result<(E::Scalar, Vec<E::Scalar>), SpartanError> {
    if self.compressed_polys.len() != num_rounds {
      return Err(SpartanError::InvalidSumcheckProof);
    }

    let mut r = Vec::with_capacity(num_rounds);
    let mut claim = initial_claim;

    for compressed in &self.compressed_polys {
      // degree check: compressed length == degree (constant + high-degree coeffs)
      if compressed.degree() != degree_bound {
        return Err(SpartanError::InvalidSumcheckProof);
      }
      let poly = compressed.decompress(&claim);

      // intra-round consistency: p(0) + p(1) == running claim
      let p0 = poly.evaluate(&E::Scalar::zero());
      let p1 = poly.evaluate(&E::Scalar::one());
      if p0 + p1 != claim {
        return Err(SpartanError::InvalidSumcheckProof);
      }

      transcript.absorb(b"p", &poly);
      let r_i = transcript.squeeze(b"c")?;
      claim = poly.evaluate(&r_i);
      r.push(r_i);
    }

    Ok((claim, r))
  }

  /// Degree-2 sumcheck on the integrand `A(x) * B(x)`. Returns
  /// `(proof, challenges, [A(r), B(r)])`.
  ///
  /// Used by the inner sumcheck of imod-Spartan.
  #[allow(non_snake_case, clippy::too_many_arguments)]
  pub fn prove_quad(
    claim: &E::Scalar,
    num_rounds: usize,
    poly_A: &mut MultilinearPolynomial<E::Scalar>,
    poly_B: &mut MultilinearPolynomial<E::Scalar>,
    transcript: &mut E::TE,
  ) -> Result<(Self, Vec<E::Scalar>, Vec<E::Scalar>), SpartanError> {
    assert_eq!(poly_A.len(), poly_B.len());
    assert_eq!(poly_A.len(), 1 << num_rounds);

    let mut r = Vec::with_capacity(num_rounds);
    let mut compressed_polys = Vec::with_capacity(num_rounds);
    let mut running_claim = *claim;
    let two = E::Scalar::from(2);

    for _ in 0..num_rounds {
      // After previous binds, poly_A has 2n entries: lower half = X=0
      // evaluations, upper half = X=1 evaluations. Compute the round
      // polynomial at X = 0, 1, 2 (3 points → degree 2).
      let n = poly_A.len() / 2;
      let mut eval_0 = E::Scalar::zero();
      let mut eval_1 = E::Scalar::zero();
      let mut eval_2 = E::Scalar::zero();

      for i in 0..n {
        let a0 = poly_A[i];
        let a1 = poly_A[n + i];
        let b0 = poly_B[i];
        let b1 = poly_B[n + i];

        eval_0 += a0 * b0;
        eval_1 += a1 * b1;

        // Linear extension to X=2: a(2) = 2*a1 - a0, same for b.
        let a2 = two * a1 - a0;
        let b2 = two * b1 - b0;
        eval_2 += a2 * b2;
      }

      debug_assert_eq!(eval_0 + eval_1, running_claim);

      let round_poly = UniPoly::from_evals(&[eval_0, eval_1, eval_2])?;
      transcript.absorb(b"p", &round_poly);
      let r_i = transcript.squeeze(b"c")?;
      compressed_polys.push(round_poly.compress());

      poly_A.bind_poly_var_top(&r_i);
      poly_B.bind_poly_var_top(&r_i);

      running_claim = round_poly.evaluate(&r_i);
      r.push(r_i);
    }

    let claims = vec![poly_A[0], poly_B[0]];
    Ok((Self::new(compressed_polys), r, claims))
  }

  /// Degree-3 sumcheck on the imod outer-SC integrand
  ///   `eq(tau, x) * (A(x) * B(x) - C(x) - M(x) * Q(x))`.
  ///
  /// Returns `(proof, challenges, [v_a, v_b, v_c, v_m, v_q])`. The
  /// initial claim is implicitly zero (the integrand vanishes on a
  /// satisfying assignment).
  #[allow(non_snake_case, clippy::too_many_arguments)]
  pub fn prove_cubic_with_five_inputs(
    claim: &E::Scalar,
    taus: Vec<E::Scalar>,
    poly_A: &mut MultilinearPolynomial<E::Scalar>,
    poly_B: &mut MultilinearPolynomial<E::Scalar>,
    poly_C: &mut MultilinearPolynomial<E::Scalar>,
    poly_M: &mut MultilinearPolynomial<E::Scalar>,
    poly_Q: &mut MultilinearPolynomial<E::Scalar>,
    transcript: &mut E::TE,
  ) -> Result<(Self, Vec<E::Scalar>, Vec<E::Scalar>), SpartanError> {
    let num_rounds = taus.len();
    let expected_len = 1usize << num_rounds;
    assert_eq!(poly_A.len(), expected_len);
    assert_eq!(poly_B.len(), expected_len);
    assert_eq!(poly_C.len(), expected_len);
    assert_eq!(poly_M.len(), expected_len);
    assert_eq!(poly_Q.len(), expected_len);

    // Build the eq table and bind it like any other MLE each round.
    // O(2^num_rounds) memory — fine for our Phase-2 sizes; revisit with
    // Gruen factoring if needed.
    let eq_evals = EqPolynomial::evals_from_points(&taus);
    let mut poly_eq = MultilinearPolynomial::new(eq_evals);

    let mut r = Vec::with_capacity(num_rounds);
    let mut compressed_polys = Vec::with_capacity(num_rounds);
    let mut running_claim = *claim;

    let two = E::Scalar::from(2);
    let three = E::Scalar::from(3);

    for _ in 0..num_rounds {
      let n = poly_A.len() / 2;
      let mut eval_0 = E::Scalar::zero();
      let mut eval_1 = E::Scalar::zero();
      let mut eval_2 = E::Scalar::zero();
      let mut eval_3 = E::Scalar::zero();

      // For each surviving variable index `i`, evaluate the integrand at
      // X = 0, 1, 2, 3 (4 points → degree 3).
      for i in 0..n {
        // Lower-half (X=0) and upper-half (X=1) evaluations
        let eq0 = poly_eq[i];
        let eq1 = poly_eq[n + i];
        let a0 = poly_A[i];
        let a1 = poly_A[n + i];
        let b0 = poly_B[i];
        let b1 = poly_B[n + i];
        let c0 = poly_C[i];
        let c1 = poly_C[n + i];
        let m0 = poly_M[i];
        let m1 = poly_M[n + i];
        let q0 = poly_Q[i];
        let q1 = poly_Q[n + i];

        // Linear extension to X=2 and X=3:
        //   p(2) = 2*p(1) - p(0)
        //   p(3) = 3*p(1) - 2*p(0)
        let eq2 = two * eq1 - eq0;
        let eq3 = three * eq1 - two * eq0;
        let a2 = two * a1 - a0;
        let a3 = three * a1 - two * a0;
        let b2 = two * b1 - b0;
        let b3 = three * b1 - two * b0;
        let c2 = two * c1 - c0;
        let c3 = three * c1 - two * c0;
        let m2 = two * m1 - m0;
        let m3 = three * m1 - two * m0;
        let q2 = two * q1 - q0;
        let q3 = three * q1 - two * q0;

        eval_0 += eq0 * (a0 * b0 - c0 - m0 * q0);
        eval_1 += eq1 * (a1 * b1 - c1 - m1 * q1);
        eval_2 += eq2 * (a2 * b2 - c2 - m2 * q2);
        eval_3 += eq3 * (a3 * b3 - c3 - m3 * q3);
      }

      debug_assert_eq!(eval_0 + eval_1, running_claim);

      let round_poly = UniPoly::from_evals(&[eval_0, eval_1, eval_2, eval_3])?;
      transcript.absorb(b"p", &round_poly);
      let r_i = transcript.squeeze(b"c")?;
      compressed_polys.push(round_poly.compress());

      poly_eq.bind_poly_var_top(&r_i);
      poly_A.bind_poly_var_top(&r_i);
      poly_B.bind_poly_var_top(&r_i);
      poly_C.bind_poly_var_top(&r_i);
      poly_M.bind_poly_var_top(&r_i);
      poly_Q.bind_poly_var_top(&r_i);

      running_claim = round_poly.evaluate(&r_i);
      r.push(r_i);
    }

    let claims = vec![poly_A[0], poly_B[0], poly_C[0], poly_M[0], poly_Q[0]];
    Ok((Self::new(compressed_polys), r, claims))
  }
}

#[cfg(test)]
mod tests {
  use super::*;
  use crate::provider::PallasHyraxEngine;
  use crate::traits::Engine;
  use ff::Field;
  use rand::SeedableRng;
  use rand::rngs::StdRng;

  type E = PallasHyraxEngine;
  type F = <E as Engine>::Scalar;

  fn rand_vec(n: usize, seed: u64) -> Vec<F> {
    let mut rng = StdRng::seed_from_u64(seed);
    (0..n).map(|_| F::random(&mut rng)).collect()
  }

  #[test]
  #[allow(non_snake_case)]
  fn prove_quad_roundtrips() {
    let num_rounds = 4usize;
    let n = 1usize << num_rounds;
    let A = rand_vec(n, 1);
    let B = rand_vec(n, 2);
    let claim: F = A.iter().zip(B.iter()).map(|(a, b)| *a * *b).sum();

    let mut poly_A = MultilinearPolynomial::new(A);
    let mut poly_B = MultilinearPolynomial::new(B);

    let mut pt = <E as Engine>::TE::new(b"test");
    let (proof, r_prover, final_evals) =
      SumcheckProof::<E>::prove_quad(&claim, num_rounds, &mut poly_A, &mut poly_B, &mut pt)
        .unwrap();

    // verifier
    let mut vt = <E as Engine>::TE::new(b"test");
    let (final_claim, r_verifier) = proof.verify(claim, num_rounds, 2, &mut vt).unwrap();

    assert_eq!(r_prover, r_verifier);
    // final claim must equal A(r) * B(r)
    assert_eq!(final_claim, final_evals[0] * final_evals[1]);
  }

  #[test]
  #[allow(non_snake_case)]
  fn prove_cubic_with_five_inputs_roundtrips_on_satisfying_witness() {
    // Construct A, B, C, M, Q so that A*B == C + M*Q pointwise.
    let num_rounds = 3usize;
    let n = 1usize << num_rounds;
    let A = rand_vec(n, 10);
    let B = rand_vec(n, 11);
    let M = rand_vec(n, 12);
    let Q = rand_vec(n, 13);
    let C: Vec<F> = (0..n).map(|i| A[i] * B[i] - M[i] * Q[i]).collect();

    let taus = rand_vec(num_rounds, 20);

    let mut poly_A = MultilinearPolynomial::new(A);
    let mut poly_B = MultilinearPolynomial::new(B);
    let mut poly_C = MultilinearPolynomial::new(C);
    let mut poly_M = MultilinearPolynomial::new(M);
    let mut poly_Q = MultilinearPolynomial::new(Q);

    let mut pt = <E as Engine>::TE::new(b"test");
    let (proof, r_prover, finals) = SumcheckProof::<E>::prove_cubic_with_five_inputs(
      &F::zero(),
      taus.clone(),
      &mut poly_A,
      &mut poly_B,
      &mut poly_C,
      &mut poly_M,
      &mut poly_Q,
      &mut pt,
    )
    .unwrap();

    // verifier
    let mut vt = <E as Engine>::TE::new(b"test");
    let (final_claim, r_verifier) = proof.verify(F::zero(), num_rounds, 3, &mut vt).unwrap();

    assert_eq!(r_prover, r_verifier);

    // reconstruct: final_claim should equal eq(tau, r) * (va*vb - vc - vm*vq)
    let (va, vb, vc, vm, vq) = (finals[0], finals[1], finals[2], finals[3], finals[4]);
    let eq_tau_r = EqPolynomial::new(taus).evaluate(&r_verifier);
    let expected = eq_tau_r * (va * vb - vc - vm * vq);
    assert_eq!(final_claim, expected);
  }

  #[test]
  #[allow(non_snake_case)]
  fn verify_rejects_tampered_proof() {
    let num_rounds = 3usize;
    let n = 1usize << num_rounds;
    let A = rand_vec(n, 30);
    let B = rand_vec(n, 31);
    let claim: F = A.iter().zip(B.iter()).map(|(a, b)| *a * *b).sum();

    let mut poly_A = MultilinearPolynomial::new(A);
    let mut poly_B = MultilinearPolynomial::new(B);

    let mut pt = <E as Engine>::TE::new(b"test");
    let (proof, _r, _finals) =
      SumcheckProof::<E>::prove_quad(&claim, num_rounds, &mut poly_A, &mut poly_B, &mut pt)
        .unwrap();

    // Tamper mode 1: drop a round so the proof has the wrong length. The
    // intra-round-consistency check would silently accept any tampering of
    // the compressed constant term (since `decompress` recovers the linear
    // coefficient from the running claim, preserving `p(0) + p(1)`), so we
    // need a tamper that breaks an invariant the verifier *actually*
    // checks. Length mismatch and degree bound are both checked.
    let mut short = proof.clone();
    short.compressed_polys.pop();
    let mut vt = <E as Engine>::TE::new(b"test");
    assert!(short.verify(claim, num_rounds, 2, &mut vt).is_err());

    // Tamper mode 2: replace a round poly with one of the wrong degree.
    let mut wrong_deg = proof;
    wrong_deg.compressed_polys[0]
      .coeffs_except_linear_term
      .pop();
    let mut vt2 = <E as Engine>::TE::new(b"test");
    assert!(wrong_deg.verify(claim, num_rounds, 2, &mut vt2).is_err());
  }
}
