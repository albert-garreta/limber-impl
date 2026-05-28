// TODO Phase 2 step 7+: when a real Mod-PCS (with DynPrime bridging) lands,
// remove `allow(dead_code)`. For step 6 this module is consumed only by its
// own tests.
#![allow(dead_code)]

//! Trivial Mod-PCS that delegates to the underlying `Engine::PCS`.
//!
//! Phase-2 step 6 scaffold: when a `ModEngine`'s `Scalar` is the same type
//! as the underlying static curve's scalar (i.e. there's no dynamic prime
//! yet), the Mod-PCS interface can be satisfied by just forwarding every
//! method to the existing `PCSEngineTrait` impl. Step 7 will add a real
//! `DynPrime`-bridging Mod-PCS that converts between the dynamic prime
//! field and the underlying curve scalar.

use crate::{
  errors::SpartanError,
  traits::{
    Engine,
    mod_engine::{ModEngine, ModPCSEngineTrait},
    pcs::PCSEngineTrait,
  },
};
use core::marker::PhantomData;

/// Mod-PCS whose implementation is "delegate everything to `E::PCS`".
/// Only usable for `ModEngine` impls where `Scalar = Engine::Scalar`
/// (the static-modulus backward-compat path).
pub struct TrivialModPCS<E: Engine> {
  _phantom: PhantomData<E>,
}

impl<E: Engine> Clone for TrivialModPCS<E> {
  fn clone(&self) -> Self {
    Self {
      _phantom: PhantomData,
    }
  }
}

// Two type parameters: `E` is the underlying static-curve `Engine`, and `M`
// is the `ModEngine` we're providing the Mod-PCS for. The bound
// `M: ModEngine<Scalar = E::Scalar, TE = E::TE>` asserts what the blanket
// `SumcheckEngine for Engine` impl already guarantees: when M and E refer to
// the same concrete engine type, their Scalar/TE are the same. Splitting
// the names avoids the "ambiguous associated type" diamond.
impl<E, M> ModPCSEngineTrait<M> for TrivialModPCS<E>
where
  E: Engine,
  M: ModEngine<Scalar = E::Scalar, TE = E::TE>,
{
  type CommitmentKey = <E::PCS as PCSEngineTrait<E>>::CommitmentKey;
  type VerifierKey = <E::PCS as PCSEngineTrait<E>>::VerifierKey;
  type Commitment = <E::PCS as PCSEngineTrait<E>>::Commitment;
  type Blind = <E::PCS as PCSEngineTrait<E>>::Blind;
  type EvaluationArgument = <E::PCS as PCSEngineTrait<E>>::EvaluationArgument;

  fn setup(
    label: &'static [u8],
    n: usize,
    width: usize,
  ) -> (Self::CommitmentKey, Self::VerifierKey) {
    E::PCS::setup(label, n, width)
  }

  fn precompute_ck(ck: &Self::CommitmentKey) {
    E::PCS::precompute_ck(ck)
  }

  fn blind(ck: &Self::CommitmentKey, n: usize) -> Self::Blind {
    E::PCS::blind(ck, n)
  }

  fn commit(
    ck: &Self::CommitmentKey,
    v: &[M::Scalar],
    r: &Self::Blind,
    is_small: bool,
  ) -> Result<Self::Commitment, SpartanError> {
    E::PCS::commit(ck, v, r, is_small)
  }

  fn check_commitment(comm: &Self::Commitment, n: usize, width: usize) -> Result<(), SpartanError> {
    E::PCS::check_commitment(comm, n, width)
  }

  fn prove(
    ck: &Self::CommitmentKey,
    ck_eval: &Self::CommitmentKey,
    transcript: &mut M::TE,
    comm: &Self::Commitment,
    poly: &[M::Scalar],
    blind: &Self::Blind,
    point: &[M::Scalar],
    comm_eval: &Self::Commitment,
    blind_eval: &Self::Blind,
  ) -> Result<Self::EvaluationArgument, SpartanError> {
    E::PCS::prove(
      ck, ck_eval, transcript, comm, poly, blind, point, comm_eval, blind_eval,
    )
  }

  fn verify(
    vk: &Self::VerifierKey,
    ck_eval: &Self::CommitmentKey,
    transcript: &mut M::TE,
    comm: &Self::Commitment,
    point: &[M::Scalar],
    comm_eval: &Self::Commitment,
    arg: &Self::EvaluationArgument,
  ) -> Result<(), SpartanError> {
    E::PCS::verify(vk, ck_eval, transcript, comm, point, comm_eval, arg)
  }
}

#[cfg(test)]
mod tests {
  use super::*;
  use crate::provider::T256HyraxEngine;
  use crate::traits::transcript::TranscriptEngineTrait;
  use ff::Field;
  use rand::SeedableRng;
  use rand::rngs::StdRng;

  type E = T256HyraxEngine;
  type F = <E as Engine>::Scalar;
  type MP = TrivialModPCS<E>;

  /// Round-trip smoke test: setup → commit → prove → verify.
  /// Exercises the full `ModPCSEngineTrait` surface end-to-end. If this
  /// compiles and passes, the trait machinery (ModEngine + ModPCSEngineTrait
  /// + transcript) is wired up correctly.
  #[test]
  fn commit_prove_verify_roundtrips() {
    let num_vars = 4usize;
    let n = 1usize << num_vars;
    let mut rng = StdRng::seed_from_u64(7);

    // Setup. Use fully qualified path on the trait because TrivialModPCS's
    // impl is generic over which ModEngine `M` it implements for.
    let (ck, vk) = <MP as ModPCSEngineTrait<E>>::setup(b"trivial-modpcs-test", n, 256);
    let (ck_eval, _) = <MP as ModPCSEngineTrait<E>>::setup(b"ck_eval", 1, 1);

    // Random polynomial + evaluation point.
    let poly: Vec<F> = (0..n).map(|_| F::random(&mut rng)).collect();
    let point: Vec<F> = (0..num_vars).map(|_| F::random(&mut rng)).collect();

    // Eval the multilinear at the point directly (oracle).
    use crate::polys_modp::multilinear::MultilinearPolynomial;
    let eval = MultilinearPolynomial::new(poly.clone(), ()).evaluate(&point);

    // Commit and prove.
    let blind = <MP as ModPCSEngineTrait<E>>::blind(&ck, n);
    let comm = <MP as ModPCSEngineTrait<E>>::commit(&ck, &poly, &blind, false).unwrap();
    let blind_eval = <MP as ModPCSEngineTrait<E>>::blind(&ck_eval, 1);
    let comm_eval =
      <MP as ModPCSEngineTrait<E>>::commit(&ck_eval, &[eval], &blind_eval, false).unwrap();

    let mut transcript_p = <E as Engine>::TE::new(b"smoke");
    let arg = <MP as ModPCSEngineTrait<E>>::prove(
      &ck,
      &ck_eval,
      &mut transcript_p,
      &comm,
      &poly,
      &blind,
      &point,
      &comm_eval,
      &blind_eval,
    )
    .unwrap();

    // Verify.
    let mut transcript_v = <E as Engine>::TE::new(b"smoke");
    <MP as ModPCSEngineTrait<E>>::verify(
      &vk,
      &ck_eval,
      &mut transcript_v,
      &comm,
      &point,
      &comm_eval,
      &arg,
    )
    .unwrap();
  }
}
