// TODO Phase 3: replace this stub with the real IntEval-based Mod-PCS
// (limb-split + range-check + IntEval). This impl is **unsound by design**
// — the verifier accepts every opening unconditionally. Exists so the
// Phase-2 SNARK driver flow can run end-to-end (sample-`p`-from-transcript,
// run sumcheck in Z_p, open at Z_p points) while the sound bridging
// protocol is still future work.
#![allow(dead_code)]

//! `TrivialIntModPCS`: a stub Mod-PCS for any `ModEngine`. Commit is a
//! Keccak hash of the polynomial bytes; prove returns an empty evaluation
//! argument; verify accepts unconditionally. Useful only for exercising
//! the driver-level plumbing in Phase 2 — not for any soundness claim.
//!
//! The commitment binds the polynomial bytes (so the verifier's
//! re-commitment of the eval value yields a determined `comm_eval`), but
//! there is no link between the polynomial commitment and the claimed
//! eval. A malicious prover can convince the verifier of false statements.

use crate::{
  errors::SpartanError,
  traits::{
    mod_engine::{ModEngine, ModPCSEngineTrait, SumcheckEngine},
    transcript::TranscriptReprTrait,
  },
};
use core::marker::PhantomData;
use num_bigint::BigUint;
use serde::{Deserialize, Serialize};
use sha3::{Digest, Keccak256};

/// Hash-based commitment to a `BigUint`-valued polynomial.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct TrivialIntCommitment {
  digest: [u8; 32],
}

impl TranscriptReprTrait for TrivialIntCommitment {
  fn to_transcript_bytes(&self) -> Vec<u8> {
    self.digest.to_vec()
  }
}

/// Trivial blind: 32 bytes mixed into the commitment digest so distinct
/// commitments of the same polynomial differ. Not hiding in any sound
/// sense — exists only to match the trait shape.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct TrivialIntBlind {
  bytes: [u8; 32],
}

/// Empty evaluation argument — the trivial verify accepts unconditionally.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct TrivialIntEvalArg;

/// Trivial key types (unit-equivalent, kept named for clarity in proofs).
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct TrivialIntKey;

/// `Phase-2` stub Mod-PCS. Unsound; exists for driver plumbing only.
pub struct TrivialIntModPCS<M: ModEngine> {
  _phantom: PhantomData<M>,
}

impl<M: ModEngine> Clone for TrivialIntModPCS<M> {
  fn clone(&self) -> Self {
    Self {
      _phantom: PhantomData,
    }
  }
}

fn hash_poly(v: &[BigUint], blind: &TrivialIntBlind) -> [u8; 32] {
  let mut h = Keccak256::new();
  h.update(b"TrivialIntModPCS/commit");
  h.update((v.len() as u64).to_le_bytes());
  for x in v {
    let bytes = x.to_bytes_le();
    h.update((bytes.len() as u64).to_le_bytes());
    h.update(&bytes);
  }
  h.update(blind.bytes);
  h.finalize().into()
}

impl<M: ModEngine> ModPCSEngineTrait<M> for TrivialIntModPCS<M> {
  type CommitmentKey = TrivialIntKey;
  type VerifierKey = TrivialIntKey;
  type Commitment = TrivialIntCommitment;
  type Blind = TrivialIntBlind;
  type EvaluationArgument = TrivialIntEvalArg;

  fn setup(
    _label: &'static [u8],
    _n: usize,
    _width: usize,
  ) -> (Self::CommitmentKey, Self::VerifierKey) {
    (TrivialIntKey, TrivialIntKey)
  }

  fn blind(_ck: &Self::CommitmentKey, _n: usize) -> Self::Blind {
    // Deterministic-zero blind keeps commit/recommit consistent without
    // needing RNG plumbing. The stub doesn't try to be hiding.
    TrivialIntBlind { bytes: [0u8; 32] }
  }

  fn commit(
    _ck: &Self::CommitmentKey,
    v: &[BigUint],
    r: &Self::Blind,
    _is_small: bool,
  ) -> Result<Self::Commitment, SpartanError> {
    Ok(TrivialIntCommitment {
      digest: hash_poly(v, r),
    })
  }

  fn check_commitment(
    _comm: &Self::Commitment,
    _n: usize,
    _width: usize,
  ) -> Result<(), SpartanError> {
    Ok(())
  }

  fn prove(
    _ck: &Self::CommitmentKey,
    _ck_eval: &Self::CommitmentKey,
    _transcript: &mut <M as SumcheckEngine>::TE,
    _comm: &Self::Commitment,
    _poly: &[BigUint],
    _blind: &Self::Blind,
    _point: &[<M as SumcheckEngine>::Scalar],
    _comm_eval: &Self::Commitment,
    _blind_eval: &Self::Blind,
  ) -> Result<Self::EvaluationArgument, SpartanError> {
    Ok(TrivialIntEvalArg)
  }

  fn verify(
    _vk: &Self::VerifierKey,
    _ck_eval: &Self::CommitmentKey,
    _transcript: &mut <M as SumcheckEngine>::TE,
    _comm: &Self::Commitment,
    _point: &[<M as SumcheckEngine>::Scalar],
    _comm_eval: &Self::Commitment,
    _arg: &Self::EvaluationArgument,
  ) -> Result<(), SpartanError> {
    Ok(())
  }
}
