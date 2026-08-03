//! Internal F-side commitment backend for the integer Mod-PCS: the seam
//! that lets the IntEval protocol run over either Pedersen/Hyrax
//! commitments (homomorphic, tiny proofs, group MSMs) or Brakedown
//! (hash-based, large proofs, no MSMs). Everything above this seam —
//! reduction sumchecks, chains, the committed-chunk representation and
//! fold points, the lockstep range check, and the interleaved
//! claim-reduction sumcheck — is backend-agnostic field arithmetic; the
//! backend only (a) commits F-polynomials and (b) discharges the final
//! one-evaluation-per-commitment openings the claim reduction produces.
//!
//! The Hyrax backend preserves the existing protocol byte-for-byte
//! (merged same-column IPA over homomorphically combined commitments,
//! Pedersen blinds). The Brakedown backend is deliberately non-hiding
//! (`Blind = ()`, no-ZK — matching the Zinc+ comparison target) and
//! opens each target with a tensor-IOPP column-opening argument.

// Staged: consumed by the Brakedown Mod-PCS integration (threading the
// backend through integer_modpcs is the next milestone).
#![allow(dead_code)]

use crate::{
  errors::SpartanError,
  provider::pcs::brakedown::{
    BrakedownCommitData, BrakedownEvalArg, BrakedownParams, brakedown_commit,
    brakedown_open_with_data, brakedown_verify_open,
  },
  provider::pt256::t256,
  traits::transcript::ByteTranscript,
};
use serde::{Serialize, de::DeserializeOwned};
use std::collections::HashMap;
use std::sync::{Mutex, OnceLock};

/// Security level for Brakedown layouts (column-open count derivation).
const BD_LAMBDA: usize = 128;
/// Public seed for the deterministic expander-code matrices; both prover
/// and verifier derive identical layouts from (length, spec, seed).
const BD_SEED: &[u8] = b"imod-modpcs-brakedown-v1";

/// Per-length Brakedown layout cache (code sampling is deterministic in
/// the public seed, so prover and verifier agree without transport).
pub(crate) fn bd_params(n: usize) -> &'static BrakedownParams<t256::Scalar> {
  static CACHE: OnceLock<Mutex<HashMap<usize, &'static BrakedownParams<t256::Scalar>>>> =
    OnceLock::new();
  let cache = CACHE.get_or_init(|| Mutex::new(HashMap::new()));
  let mut guard = cache.lock().expect("bd params cache poisoned");
  if let Some(p) = guard.get(&n) {
    return p;
  }
  let params = Box::leak(Box::new(BrakedownParams::new(
    n,
    crate::provider::pcs::brakedown::DEFAULT_SPEC,
    BD_LAMBDA,
    BD_SEED,
  )));
  guard.insert(n, params);
  params
}

/// One final-opening target after the interleaved claim reduction: the
/// commitment, its polynomial (and any retained commit data), the
/// reduced point, and the claimed evaluation (already transcript-bound
/// by the claim reduction).
pub struct OpenTarget<'a, B: FBackend> {
  pub comm: &'a B::Comm,
  pub poly: &'a [t256::Scalar],
  pub blind: &'a B::Blind,
  pub data: &'a B::Data,
  pub point: Vec<t256::Scalar>,
  pub eval: t256::Scalar,
}

/// The F-side commitment backend seam. `Data` is whatever the prover
/// retains alongside a commitment to answer openings (Hyrax: the blind;
/// Brakedown: the encoded matrix + Merkle tree).
pub trait FBackend: Sized + Send + Sync + 'static {
  type Ck: Send + Sync + Clone;
  type Vk: Send + Sync + Clone;
  type Comm: Clone + core::fmt::Debug + PartialEq + Serialize + DeserializeOwned + Send + Sync;
  type Blind: Clone + core::fmt::Debug + Serialize + DeserializeOwned + Send + Sync;
  type Data: Send + Sync;
  type BatchOpenArg: Clone + core::fmt::Debug + Serialize + DeserializeOwned + Send + Sync;

  /// Fresh commitment randomness for an `n`-coefficient polynomial
  /// (`()` for non-hiding backends).
  fn blind(ck: &Self::Ck, n: usize) -> Self::Blind;

  /// Transcript representation of a commitment (`absorb`-equivalent).
  fn comm_transcript_bytes(comm: &Self::Comm) -> Vec<u8>;

  /// Commit to an F-polynomial under `blind`. `small` hints that every
  /// coefficient is < 2^16 (Hyrax small-scalar MSM path; Brakedown
  /// ignores it). Deterministic given `(poly, blind)` — callers
  /// recommit to check commitment equality.
  fn commit(
    ck: &Self::Ck,
    poly: &[t256::Scalar],
    blind: &Self::Blind,
    small: bool,
  ) -> Result<(Self::Comm, Self::Data), SpartanError>;

  /// Discharge every target's single-point evaluation claim against a
  /// shared sub-transcript (the claims and evals are already bound to it
  /// by the claim-reduction phase).
  fn open_targets(
    ck: &Self::Ck,
    targets: &[OpenTarget<'_, Self>],
    sub: &mut impl ByteTranscript,
  ) -> Result<Self::BatchOpenArg, SpartanError>;

  /// Verifier mirror of [`Self::open_targets`].
  fn verify_targets(
    vk: &Self::Vk,
    targets: &[(&Self::Comm, Vec<t256::Scalar>, t256::Scalar)],
    arg: &Self::BatchOpenArg,
    sub: &mut impl ByteTranscript,
  ) -> Result<(), SpartanError>;
}

/// Brakedown (hash-based) backend: non-hiding, MSM-free, per-target
/// tensor-IOPP openings. The comparison instantiation.
#[derive(Clone, Debug)]
pub struct BdBackend;

impl FBackend for BdBackend {
  type Ck = ();
  type Vk = ();
  type Comm = [u8; 32];
  type Blind = ();
  type Data = BrakedownCommitData<t256::Scalar>;
  type BatchOpenArg = Vec<BrakedownEvalArg<t256::Scalar>>;

  fn blind(_ck: &Self::Ck, _n: usize) -> Self::Blind {}

  fn comm_transcript_bytes(comm: &Self::Comm) -> Vec<u8> {
    comm.to_vec()
  }

  fn commit(
    _ck: &Self::Ck,
    poly: &[t256::Scalar],
    _blind: &Self::Blind,
    _small: bool,
  ) -> Result<(Self::Comm, Self::Data), SpartanError> {
    let params = bd_params(poly.len().next_power_of_two());
    let mut padded;
    let poly = if poly.len() == params.poly_len() {
      poly
    } else {
      padded = poly.to_vec();
      padded.resize(params.poly_len(), t256::Scalar::from(0u64));
      &padded[..]
    };
    let (root, data) = brakedown_commit(params, poly);
    Ok((root, data))
  }

  fn open_targets(
    _ck: &Self::Ck,
    targets: &[OpenTarget<'_, Self>],
    sub: &mut impl ByteTranscript,
  ) -> Result<Self::BatchOpenArg, SpartanError> {
    let mut args = Vec::with_capacity(targets.len());
    for t in targets {
      let params = bd_params(t.poly.len().next_power_of_two());
      let (eval, arg) = brakedown_open_with_data(params, t.comm, t.data, &t.point, sub)?;
      debug_assert_eq!(eval, t.eval, "claim reduction / opening eval mismatch");
      args.push(arg);
    }
    Ok(args)
  }

  fn verify_targets(
    _vk: &Self::Vk,
    targets: &[(&Self::Comm, Vec<t256::Scalar>, t256::Scalar)],
    arg: &Self::BatchOpenArg,
    sub: &mut impl ByteTranscript,
  ) -> Result<(), SpartanError> {
    if arg.len() != targets.len() {
      return Err(SpartanError::ProofVerifyError {
        reason: "brakedown backend: wrong number of opening arguments".to_string(),
      });
    }
    for ((comm, point, eval), a) in targets.iter().zip(arg.iter()) {
      let params = bd_params(1usize << point.len());
      brakedown_verify_open(params, comm, point, *eval, a, sub)?;
    }
    Ok(())
  }
}
