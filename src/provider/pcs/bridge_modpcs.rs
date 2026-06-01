// TODO Phase 3: this is the Mod-PCS *compiler skeleton* specialized to
// `numlimb = 1` (p = q, witness values fit in a single field-sized limb).
// Phase 3 generalizes the body to `numlimb > 1`: limb-split the
// non-negative bounded-integer witness, range-check each limb in [0, B),
// run IntEval. (We assume non-negative witnesses for this implementation,
// so unsigned limb-splitting and a one-sided range check suffice — no
// signed/centered handling.) The interface (commit bounded values, open at
// Z_p points) stays; only the body changes. `allow(dead_code)` until a
// Phase-2 SNARK driver consumes it.
#![allow(dead_code)]

//! `BridgeModPCS`: a Mod-PCS for `T256DynPrimeEngine` that delegates to the
//! Hyrax PCS over T256, under the `p = q` shortcut (the dynamic prime equals
//! the curve scalar prime, so `DynPrime<4> ↔ t256::Scalar` is a
//! value-preserving cast and `numlimb = 1`).
//!
//! Concrete (not generic) on purpose: it pins the `(DynPrime<4>, t256::Scalar)`
//! conversion rather than fighting a generic `DynPrime<L> → E::Scalar` bound.
//! Thanks to the field-agnostic `ByteTranscript`, the underlying Hyrax
//! opening runs on the shared transcript with no fork/reinterpret — we pass
//! the DynPrime transcript straight through.

use crate::{
  dyn_prime::DynPrime,
  errors::SpartanError,
  provider::{T256DynPrimeEngine, T256HyraxEngine, pcs::hyrax_pc::HyraxPCS, pt256::t256},
  traits::{
    PrimeFieldExt,
    mod_engine::{ModPCSEngineTrait, SumcheckEngine, SumcheckField},
    pcs::PCSEngineTrait,
  },
};
use ff::PrimeField;
use num_bigint::BigUint;

/// Underlying standard PCS: Hyrax over T256.
type Hyrax = HyraxPCS<T256HyraxEngine>;

/// Convert a `DynPrime<4>` (modulus = T256 scalar prime) into the T256
/// curve scalar. Value-preserving because the two share the same modulus
/// (`p = q`); `retrieve()` yields the canonical integer `< p = < q`, which
/// is a valid canonical `t256::Scalar` representation.
fn dyn_to_scalar(d: &DynPrime<4>) -> Result<t256::Scalar, SpartanError> {
  let le = d.to_le_bytes();
  let mut repr = <t256::Scalar as PrimeField>::Repr::default();
  if repr.as_ref().len() != le.len() {
    return Err(SpartanError::InternalError {
      reason: format!(
        "DynPrime->Scalar byte length mismatch: repr {}, dynprime {}",
        repr.as_ref().len(),
        le.len()
      ),
    });
  }
  repr.as_mut().copy_from_slice(&le);
  Option::from(t256::Scalar::from_repr(repr)).ok_or(SpartanError::InternalError {
    reason: "DynPrime value is not a canonical t256::Scalar (p != q?)".to_string(),
  })
}

fn dyn_vec_to_scalars(v: &[DynPrime<4>]) -> Result<Vec<t256::Scalar>, SpartanError> {
  v.iter().map(dyn_to_scalar).collect()
}

/// `BigUint → t256::Scalar` via 64-byte wide reduction. Value-preserving
/// for inputs that fit in the scalar field, otherwise reduces uniformly.
/// Same convention as `TrivialModPCS` — Phase 3 will swap this for limb-
/// split + range-checked commitments.
fn biguint_to_t256_scalar(v: &BigUint) -> t256::Scalar {
  let mut bytes = v.to_bytes_le();
  bytes.resize(64, 0);
  <t256::Scalar as PrimeFieldExt>::from_uniform(&bytes)
}

/// Mod-PCS for `T256DynPrimeEngine`, delegating to Hyrax-over-T256.
#[derive(Clone)]
pub struct BridgeModPCS;

impl ModPCSEngineTrait<T256DynPrimeEngine> for BridgeModPCS {
  type CommitmentKey = <Hyrax as PCSEngineTrait<T256HyraxEngine>>::CommitmentKey;
  type VerifierKey = <Hyrax as PCSEngineTrait<T256HyraxEngine>>::VerifierKey;
  type Commitment = <Hyrax as PCSEngineTrait<T256HyraxEngine>>::Commitment;
  type Blind = <Hyrax as PCSEngineTrait<T256HyraxEngine>>::Blind;
  type EvaluationArgument = <Hyrax as PCSEngineTrait<T256HyraxEngine>>::EvaluationArgument;

  fn setup(
    label: &'static [u8],
    n: usize,
    width: usize,
  ) -> (Self::CommitmentKey, Self::VerifierKey) {
    // Phase 3: also take the limb bound B and coeff bound B_f here, and
    // derive numlimb. For numlimb = 1 there's nothing extra to set up.
    Hyrax::setup(label, n, width)
  }

  fn precompute_ck(ck: &Self::CommitmentKey) {
    Hyrax::precompute_ck(ck)
  }

  fn blind(ck: &Self::CommitmentKey, n: usize) -> Self::Blind {
    Hyrax::blind(ck, n)
  }

  fn commit(
    ck: &Self::CommitmentKey,
    v: &[BigUint],
    r: &Self::Blind,
    is_small: bool,
  ) -> Result<Self::Commitment, SpartanError> {
    // NOTE: `v` carries bounded-integer-valued BigUint entries. Under
    // numlimb = 1 each fits in the curve scalar, so we just cast and
    // commit. (Phase 3: limb-split + range-check here.)
    let v_fq: Vec<t256::Scalar> = v.iter().map(biguint_to_t256_scalar).collect();
    Hyrax::commit(ck, &v_fq, r, is_small)
  }

  fn check_commitment(comm: &Self::Commitment, n: usize, width: usize) -> Result<(), SpartanError> {
    Hyrax::check_commitment(comm, n, width)
  }

  fn prove(
    ck: &Self::CommitmentKey,
    ck_eval: &Self::CommitmentKey,
    transcript: &mut <T256DynPrimeEngine as SumcheckEngine>::TE,
    comm: &Self::Commitment,
    poly: &[BigUint],
    blind: &Self::Blind,
    point: &[<T256DynPrimeEngine as SumcheckEngine>::Scalar],
    comm_eval: &Self::Commitment,
    blind_eval: &Self::Blind,
  ) -> Result<Self::EvaluationArgument, SpartanError> {
    let poly_fq: Vec<t256::Scalar> = poly.iter().map(biguint_to_t256_scalar).collect();
    let point_fq = dyn_vec_to_scalars(point)?;
    // The shared (DynPrime) transcript is a `ByteTranscript`, which is
    // exactly what `Hyrax::prove` now accepts — no fork/reinterpret.
    Hyrax::prove(
      ck, ck_eval, transcript, comm, &poly_fq, blind, &point_fq, comm_eval, blind_eval,
    )
  }

  fn verify(
    vk: &Self::VerifierKey,
    ck_eval: &Self::CommitmentKey,
    transcript: &mut <T256DynPrimeEngine as SumcheckEngine>::TE,
    comm: &Self::Commitment,
    point: &[<T256DynPrimeEngine as SumcheckEngine>::Scalar],
    comm_eval: &Self::Commitment,
    arg: &Self::EvaluationArgument,
  ) -> Result<(), SpartanError> {
    let point_fq = dyn_vec_to_scalars(point)?;
    Hyrax::verify(vk, ck_eval, transcript, comm, &point_fq, comm_eval, arg)
  }
}

/// The T256 scalar field's prime as `FixedMontyParams<4>`, for `p = q`
/// prototyping. Computed as `(q-1) + 1` from `-Scalar::ONE` to avoid
/// parsing `PrimeField::MODULUS` string formats.
pub fn t256_scalar_params() -> crypto_bigint::modular::FixedMontyParams<4> {
  use crypto_bigint::{Odd, U256};
  let q_minus_1 = (-<t256::Scalar as ff::Field>::ONE).to_repr();
  let mut bytes = q_minus_1.as_ref().to_vec();
  let mut carry = 1u8;
  for b in bytes.iter_mut() {
    let (v, c) = b.overflowing_add(carry);
    *b = v;
    carry = u8::from(c);
  }
  debug_assert_eq!(carry, 0, "addition carried out of the modulus width");
  let modulus = U256::from_le_slice(&bytes);
  crypto_bigint::modular::FixedMontyParams::new(Odd::new(modulus).unwrap())
}

#[cfg(test)]
mod tests {
  use super::*;
  use crate::polys_modp::multilinear::MultilinearPolynomial;
  use crate::traits::transcript::TranscriptEngineTrait;

  type ME = T256DynPrimeEngine;
  type DP = DynPrime<4>;
  type MP = BridgeModPCS;

  /// Full Mod-PCS round-trip over DynPrime: setup → commit → MLE-evaluate →
  /// commit eval → prove → verify, with `p = q`. The Hyrax opening runs on
  /// the DynPrime transcript directly (ByteTranscript), and the cast
  /// `DynPrime → t256::Scalar` is value-preserving.
  #[test]
  fn commit_prove_verify_roundtrips_dynprime() {
    let num_vars = 4usize;
    let n = 1usize << num_vars;
    let params = t256_scalar_params();

    let (ck, vk) = <MP as ModPCSEngineTrait<ME>>::setup(b"bridge-test", n, 256);
    let (ck_eval, _) = <MP as ModPCSEngineTrait<ME>>::setup(b"ck_eval", 1, 1);

    let poly: Vec<DP> = (0..n)
      .map(|i| DP::from_u64(&params, (i as u64) * 37 + 5))
      .collect();
    let point: Vec<DP> = (0..num_vars)
      .map(|i| DP::from_u64(&params, (i as u64) + 3))
      .collect();

    // Oracle evaluation in DynPrime.
    let eval = MultilinearPolynomial::new(poly.clone(), params).evaluate(&point);

    // BigUint views for the commit/prove calls — Mod-PCS commits integers.
    let to_bu = |x: &DP| num_bigint::BigUint::from_bytes_le(&x.to_le_bytes());
    let poly_bu: Vec<BigUint> = poly.iter().map(to_bu).collect();

    let blind = <MP as ModPCSEngineTrait<ME>>::blind(&ck, n);
    let comm = <MP as ModPCSEngineTrait<ME>>::commit(&ck, &poly_bu, &blind, false).unwrap();
    let blind_eval = <MP as ModPCSEngineTrait<ME>>::blind(&ck_eval, 1);
    let comm_eval =
      <MP as ModPCSEngineTrait<ME>>::commit(&ck_eval, &[to_bu(&eval)], &blind_eval, false).unwrap();

    let mut pt = <ME as SumcheckEngine>::TE::new_with_params(b"smoke", params);
    let arg = <MP as ModPCSEngineTrait<ME>>::prove(
      &ck,
      &ck_eval,
      &mut pt,
      &comm,
      &poly_bu,
      &blind,
      &point,
      &comm_eval,
      &blind_eval,
    )
    .unwrap();

    let mut vt = <ME as SumcheckEngine>::TE::new_with_params(b"smoke", params);
    <MP as ModPCSEngineTrait<ME>>::verify(&vk, &ck_eval, &mut vt, &comm, &point, &comm_eval, &arg)
      .unwrap();
  }
}
