//! The interface separating the integer Mod-PCS *protocol* from the
//! *commitment scheme* it runs on.
//!
//! The protocol — sumchecks, the per-prime chains, the range check —
//! is ordinary field arithmetic and never manipulates commitments
//! algebraically. It needs a commitment scheme for exactly two things:
//!
//! 1. committing to polynomials over the Hyrax scalar field, and
//! 2. proving, at the very end, what each committed polynomial
//!    evaluates to at one point (the protocol has already reduced all
//!    of its claims down to one evaluation per commitment).
//!
//! [`CommitBackend`] captures those two operations. Two
//! implementations exist: the Pedersen/Hyrax one (`HyBackend`, in
//! `integer_modpcs.rs`), which keeps the existing protocol
//! byte-for-byte — including its trick of merging all the final
//! evaluation proofs into a single inner-product argument, which works
//! because Pedersen commitments can be added together — and the
//! hash-based Brakedown one ([`BdBackend`]), which cannot merge
//! commitments and instead proves each evaluation separately with
//! Merkle-tree column openings. Brakedown commitments have no
//! randomness, so that instantiation is not zero-knowledge (the same
//! trade-off Zinc+ makes).

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
  traits::PrimeFieldExt,
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

/// Bounded cache of commit-time opening data, keyed by Merkle root, so
/// `prove` does not re-encode witness polynomials committed moments
/// earlier through the ModPCS surface. Purely a prover-side
/// memoization: on a miss the data is recomputed and checked against
/// the expected root.
fn bd_data_cache_put(root: [u8; 32], data: BrakedownCommitData<t256::Scalar>) {
  let cache = bd_data_cache();
  let mut guard = cache.lock().expect("bd data cache poisoned");
  if guard.len() >= 8 {
    guard.clear();
  }
  guard.insert(root, data);
}

fn bd_data_cache_get(root: &[u8; 32]) -> Option<BrakedownCommitData<t256::Scalar>> {
  bd_data_cache()
    .lock()
    .expect("bd data cache poisoned")
    .get(root)
    .cloned()
}

#[allow(clippy::type_complexity)]
fn bd_data_cache() -> &'static Mutex<HashMap<[u8; 32], BrakedownCommitData<t256::Scalar>>> {
  static CACHE: OnceLock<Mutex<HashMap<[u8; 32], BrakedownCommitData<t256::Scalar>>>> =
    OnceLock::new();
  CACHE.get_or_init(|| Mutex::new(HashMap::new()))
}

/// One final-opening target after the interleaved claim reduction: the
/// commitment, its polynomial (and any retained commit data), the
/// reduced point, and the claimed evaluation (already transcript-bound
/// by the claim reduction).
pub struct OpenTarget<'a, B: CommitBackend> {
  pub comm: &'a B::Comm,
  pub poly: &'a [B::Scalar],
  pub blind: &'a B::Blind,
  pub data: &'a B::Data,
  pub point: Vec<B::Scalar>,
  pub eval: B::Scalar,
}

/// The F-side commitment backend seam. `Data` is whatever the prover
/// retains alongside a commitment to answer openings (Hyrax: the blind;
/// Brakedown: the encoded matrix + Merkle tree).
pub trait CommitBackend: Sized + Send + Sync + 'static {
  /// The prime field the committed polynomials live over (the q-side
  /// field). Today both backends set this to `t256::Scalar`; making it
  /// an associated type is what lets the same protocol instantiate
  /// over a smaller field (e.g. M127) without touching protocol code.
  type Scalar: PrimeFieldExt + Serialize + DeserializeOwned + Send + Sync + 'static;
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

  /// Regenerate the retained opening data for a polynomial that was
  /// committed elsewhere (the ModPCS commit surface returns only the
  /// commitment). `comm` is the expected commitment. Free for Hyrax
  /// (`()`); Brakedown serves it from a small cache filled at commit
  /// time, re-encoding only on a miss.
  fn recommit_data(
    ck: &Self::Ck,
    comm: &Self::Comm,
    poly: &[Self::Scalar],
    blind: &Self::Blind,
    small: bool,
  ) -> Result<Self::Data, SpartanError>;

  /// Commit to an F-polynomial under `blind`. `small` hints that every
  /// coefficient is < 2^16 (Hyrax small-scalar MSM path; Brakedown
  /// ignores it). Deterministic given `(poly, blind)` — callers
  /// recommit to check commitment equality.
  fn commit(
    ck: &Self::Ck,
    poly: &[Self::Scalar],
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
    targets: &[(&Self::Comm, Vec<Self::Scalar>, Self::Scalar)],
    arg: &Self::BatchOpenArg,
    sub: &mut impl ByteTranscript,
  ) -> Result<(), SpartanError>;
}

/// Brakedown (hash-based) backend: non-hiding, MSM-free, per-target
/// tensor-IOPP openings. The comparison instantiation.
#[derive(Clone, Debug)]
pub struct BdBackend;

impl CommitBackend for BdBackend {
  type Scalar = t256::Scalar;
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
    bd_data_cache_put(root, data.clone());
    Ok((root, data))
  }

  fn recommit_data(
    ck: &Self::Ck,
    comm: &Self::Comm,
    poly: &[t256::Scalar],
    blind: &Self::Blind,
    small: bool,
  ) -> Result<Self::Data, SpartanError> {
    if let Some(data) = bd_data_cache_get(comm) {
      return Ok(data);
    }
    let (root, data) = Self::commit(ck, poly, blind, small)?;
    if &root != comm {
      return Err(SpartanError::InternalError {
        reason: "brakedown recommit: root mismatch with the input commitment".to_string(),
      });
    }
    Ok(data)
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
