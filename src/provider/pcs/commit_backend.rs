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
    BrakedownCommitData, BrakedownDirectOpen, BrakedownGroupArg, BrakedownParams, brakedown_commit,
    brakedown_commit_plain, brakedown_open_direct, brakedown_open_group, brakedown_verify_direct,
    brakedown_verify_group,
  },
  traits::PrimeFieldExt,
  traits::transcript::ByteTranscript,
};
use serde::{Serialize, de::DeserializeOwned};
use std::collections::HashMap;
use std::sync::{Mutex, OnceLock};

/// Security level for Brakedown layouts (column-open count derivation).
/// Aligned with the accepted system floor (~114-117 bits: the 2^-114
/// fingerprint term and LAMBDA_BOUND2 = 117) rather than over-securing
/// this one term to 128 - the same argument as the challenge bound.
const BD_LAMBDA: usize = 117;
/// Public seed for the deterministic expander-code matrices; both prover
/// and verifier derive identical layouts from (length, spec, seed).
const BD_SEED: &[u8] = b"imod-modpcs-brakedown-v1";
/// Targets of at most this many coefficients ship their (compact)
/// polynomial directly instead of a column-opening argument: below this
/// size the plaintext is smaller than the argument, and the verifier
/// just recommits and evaluates. BDDIRECT overrides (benching knob).
const BD_DIRECT_MAX: usize = 1 << 16;

fn bd_direct_max() -> usize {
  std::env::var("BDDIRECT")
    .ok()
    .and_then(|v| v.parse::<usize>().ok())
    .unwrap_or(BD_DIRECT_MAX)
}

/// The Brakedown backend's batch-opening argument: column-opening
/// groups for the large targets (in group order) and directly-shipped
/// polynomials for the small ones (in canonical target order). The
/// partition and grouping are deterministic functions of the target
/// list, so the verifier reconstructs them without transport.
#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
#[serde(bound(
  serialize = "F: serde::Serialize + ff::PrimeField",
  deserialize = "F: serde::de::DeserializeOwned + ff::PrimeField"
))]
pub struct BdBatchOpenArg<F> {
  /// Grouped column-opening arguments for targets above the direct-ship
  /// threshold.
  pub groups: Vec<BrakedownGroupArg<F>>,
  /// Directly-shipped small polynomials.
  pub direct: Vec<BrakedownDirectOpen<F>>,
}

/// Per-length Brakedown layout cache (code sampling is deterministic in
/// the public seed, so prover and verifier agree without transport).
pub(crate) fn bd_params<F: crate::traits::PrimeFieldExt>(n: usize) -> &'static BrakedownParams<F> {
  use std::any::{Any, TypeId};
  // One cache across all field instantiations, keyed by (field, length);
  // generic statics are unavailable, so entries go through `dyn Any`.
  static CACHE: OnceLock<Mutex<HashMap<(TypeId, usize), &'static (dyn Any + Send + Sync)>>> =
    OnceLock::new();
  let cache = CACHE.get_or_init(|| Mutex::new(HashMap::new()));
  let mut guard = cache.lock().expect("bd params cache poisoned");
  let key = (TypeId::of::<F>(), n);
  if let Some(p) = guard.get(&key) {
    return p.downcast_ref::<BrakedownParams<F>>().expect("cache type");
  }
  // BDSPEC=<0..5> overrides the code spec (benching knob; prover and
  // verifier share the process, so layouts agree).
  let spec = std::env::var("BDSPEC")
    .ok()
    .and_then(|v| v.parse::<usize>().ok())
    .map(|i| crate::provider::pcs::brakedown::SPECS[i])
    // Default for the Mod-PCS backend: spec4 — after the 2026-08-24
    // delayed-reduction encoder win the prover margin comfortably
    // absorbs the denser code (~+35 ms for -0.7 MB proof vs spec3;
    // spec5 buys only -0.15 MB more for another ~+80 ms).
    .unwrap_or(crate::provider::pcs::brakedown::SPECS[4]);
  // Uniform row length across all lengths (BDROWLEN override, benching
  // knob). Trees sharing (row_len, spec, seed) share the code, so the
  // batch opening combines every tree's proximity/evaluation rows into
  // ONE global pair — the per-tree layout optimum is worse than a
  // shared-row global layout once rows are amortized across trees. The
  // default 2^15 sits at the global optimum L* ≈ √(t·Σn/2) for the
  // MultiSwap-scale target set and won the 2026-08-21 sweep against
  // 2^16 on proof size, prover, AND verify.
  let row_len = std::env::var("BDROWLEN")
    .ok()
    .and_then(|v| v.parse::<usize>().ok())
    .unwrap_or(1 << 15)
    .min(n);
  let params: &'static BrakedownParams<F> = Box::leak(Box::new(BrakedownParams::new_with_row_len(
    n, spec, BD_LAMBDA, BD_SEED, row_len,
  )));
  guard.insert(key, params as &'static (dyn Any + Send + Sync));
  params
}

/// Bounded cache of commit-time opening data, keyed by Merkle root, so
/// `prove` does not re-encode witness polynomials committed moments
/// earlier through the ModPCS surface. Purely a prover-side
/// memoization: on a miss the data is recomputed and checked against
/// the expected root.
fn bd_data_cache_put<F: crate::traits::PrimeFieldExt>(
  root: [u8; 32],
  data: BrakedownCommitData<F>,
) {
  let cache = bd_data_cache();
  let mut guard = cache.lock().expect("bd data cache poisoned");
  if guard.len() >= 8 {
    guard.clear();
  }
  guard.insert(
    (std::any::TypeId::of::<F>(), root),
    Box::new(data) as Box<dyn std::any::Any + Send + Sync>,
  );
}

fn bd_data_cache_get<F: crate::traits::PrimeFieldExt>(
  root: &[u8; 32],
) -> Option<BrakedownCommitData<F>> {
  bd_data_cache()
    .lock()
    .expect("bd data cache poisoned")
    .get(&(std::any::TypeId::of::<F>(), *root))
    .and_then(|b| b.downcast_ref::<BrakedownCommitData<F>>().cloned())
}

#[allow(clippy::type_complexity)]
fn bd_data_cache()
-> &'static Mutex<HashMap<(std::any::TypeId, [u8; 32]), Box<dyn std::any::Any + Send + Sync>>> {
  static CACHE: OnceLock<
    Mutex<HashMap<(std::any::TypeId, [u8; 32]), Box<dyn std::any::Any + Send + Sync>>>,
  > = OnceLock::new();
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
  type Scalar: PrimeFieldExt
    + crate::traits::transcript::TranscriptReprTrait
    + Serialize
    + DeserializeOwned
    + Send
    + Sync
    + 'static;
  /// A curve-free engine naming this backend's (Scalar, transcript)
  /// pair, for the sub-protocols (LogUp-GKR) that are generic over
  /// `SumcheckEngine`. Full `Engine`s satisfy this via the blanket
  /// impl; a small-field backend mints a ~20-line struct.
  type SE: crate::traits::mod_engine::SumcheckEngine<Scalar = Self::Scalar>;
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
pub struct BdBackend<SE = crate::provider::T256HyraxEngine>(core::marker::PhantomData<SE>);

impl<SE> CommitBackend for BdBackend<SE>
where
  SE: crate::traits::mod_engine::SumcheckEngine,
  SE::Scalar: PrimeFieldExt
    + crate::traits::transcript::TranscriptReprTrait
    + Serialize
    + DeserializeOwned
    + Send
    + Sync
    + 'static,
{
  type Scalar = SE::Scalar;
  type SE = SE;
  type Ck = ();
  type Vk = ();
  type Comm = [u8; 32];
  type Blind = ();
  type Data = BrakedownCommitData<SE::Scalar>;
  type BatchOpenArg = BdBatchOpenArg<SE::Scalar>;

  fn blind(_ck: &Self::Ck, _n: usize) -> Self::Blind {}

  fn comm_transcript_bytes(comm: &Self::Comm) -> Vec<u8> {
    comm.to_vec()
  }

  fn commit(
    _ck: &Self::Ck,
    poly: &[Self::Scalar],
    _blind: &Self::Blind,
    _small: bool,
  ) -> Result<(Self::Comm, Self::Data), SpartanError> {
    let params = bd_params(poly.len().next_power_of_two());
    let mut padded;
    let poly = if poly.len() == params.poly_len() {
      poly
    } else {
      padded = poly.to_vec();
      padded.resize(params.poly_len(), Self::Scalar::from(0u64));
      &padded[..]
    };
    // Below the direct-ship threshold the opening ships the polynomial
    // itself, so the commitment is just a plain hash of its canonical
    // bytes — no encoding, no Merkle tree (see `open_targets`).
    let (root, data) = if poly.len() <= bd_direct_max() {
      brakedown_commit_plain(poly)
    } else {
      brakedown_commit(params, poly)
    };
    bd_data_cache_put(root, data.clone());
    Ok((root, data))
  }

  fn recommit_data(
    ck: &Self::Ck,
    comm: &Self::Comm,
    poly: &[Self::Scalar],
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
    // Small targets ship their polynomial directly. The rest: targets
    // whose layouts share a code (uniform row length) and whose points
    // share their column suffix form ONE group with ONE proximity row
    // and ONE gamma-combined evaluation row (per-target rows dominated
    // the proof); with the uniform-row-length policy and the claim
    // reduction's shared challenge tail, that is normally every large
    // target. Both the partition and the grouping are by first
    // appearance in the canonical target order, so the verifier
    // reconstructs them deterministically.
    let direct_max = bd_direct_max();
    let mut direct = Vec::new();
    let mut groups: Vec<(usize, Vec<usize>)> = Vec::new();
    for (i, t) in targets.iter().enumerate() {
      if 1usize << t.point.len() <= direct_max {
        direct.push(brakedown_open_direct(t.poly));
        continue;
      }
      let params = bd_params::<SE::Scalar>(1usize << t.point.len());
      let lc = params.row_len.trailing_zeros() as usize;
      match groups.iter_mut().find(|(rep, _)| {
        let rp = &targets[*rep].point;
        bd_params::<SE::Scalar>(1usize << rp.len()).row_len == params.row_len
          && rp[rp.len() - lc..] == t.point[t.point.len() - lc..]
      }) {
        Some((_, members)) => members.push(i),
        None => groups.push((i, vec![i])),
      }
    }
    let mut args = Vec::with_capacity(groups.len());
    for (rep, members) in &groups {
      let params = bd_params(1usize << targets[*rep].point.len());
      let items: Vec<(&[u8; 32], &BrakedownCommitData<SE::Scalar>, &[SE::Scalar])> = members
        .iter()
        .map(|&i| {
          (
            targets[i].comm,
            targets[i].data,
            targets[i].point.as_slice(),
          )
        })
        .collect();
      let (evals, arg) = brakedown_open_group(params, &items, sub)?;
      for (k, &i) in members.iter().enumerate() {
        debug_assert_eq!(
          evals[k], targets[i].eval,
          "claim reduction / opening eval mismatch"
        );
        let _ = (k, i);
      }
      args.push(arg);
    }
    Ok(BdBatchOpenArg {
      groups: args,
      direct,
    })
  }

  fn verify_targets(
    _vk: &Self::Vk,
    targets: &[(&Self::Comm, Vec<Self::Scalar>, Self::Scalar)],
    arg: &Self::BatchOpenArg,
    sub: &mut impl ByteTranscript,
  ) -> Result<(), SpartanError> {
    // Reconstruct the prover's direct/grouped partition and grouping
    // (shared code + shared column suffix) from the canonical target
    // order.
    let direct_max = bd_direct_max();
    let mut n_direct = 0usize;
    let mut groups: Vec<(usize, Vec<usize>)> = Vec::new();
    for (i, (comm, point, eval)) in targets.iter().enumerate() {
      if 1usize << point.len() <= direct_max {
        let a = arg
          .direct
          .get(n_direct)
          .ok_or_else(|| SpartanError::ProofVerifyError {
            reason: "brakedown backend: missing direct opening".to_string(),
          })?;
        brakedown_verify_direct(comm, point, *eval, a)?;
        n_direct += 1;
        continue;
      }
      let params = bd_params::<SE::Scalar>(1usize << point.len());
      let lc = params.row_len.trailing_zeros() as usize;
      match groups.iter_mut().find(|(rep, _)| {
        let rp = &targets[*rep].1;
        bd_params::<SE::Scalar>(1usize << rp.len()).row_len == params.row_len
          && rp[rp.len() - lc..] == point[point.len() - lc..]
      }) {
        Some((_, members)) => members.push(i),
        None => groups.push((i, vec![i])),
      }
    }
    if arg.groups.len() != groups.len() || arg.direct.len() != n_direct {
      return Err(SpartanError::ProofVerifyError {
        reason: "brakedown backend: wrong number of opening arguments".to_string(),
      });
    }
    for ((rep, members), a) in groups.iter().zip(arg.groups.iter()) {
      let params = bd_params(1usize << targets[*rep].1.len());
      let items: Vec<(&[u8; 32], &[SE::Scalar], SE::Scalar)> = members
        .iter()
        .map(|&i| (targets[i].0, targets[i].1.as_slice(), targets[i].2))
        .collect();
      brakedown_verify_group(params, &items, a, sub)?;
    }
    Ok(())
  }
}
