// TODO Phase 3: this file is the in-progress sound Mod-PCS for
// `T256DynPrimeEngine`. Implements the IntEval protocol (Section 4 of
// the SNARKs-for-Integers paper) over Hyrax-T256 as the underlying F PCS.
// "No limb-splitting" simplification: the witness is assumed bounded by
// `T_f` and cast directly to F (range check on commit is a followup);
// `f_limb = f` so the Mod-PCS eval's reduction sumcheck is trivial.
//
// Landing in stages:
//   - step A (this commit): skeleton, params, setup, commit (cast +
//     Hyrax::commit, range check TODO), open (Hyrax delegate).
//   - step B: small-prime transcript sampling + n ≤ k IntEval (no
//     partial-evaluation iteration).
//   - step C: partial-evaluation iteration for n > k.
//   - step D: batch range check.
#![allow(dead_code)]

//! `IntEvalModPCS`: sound Mod-PCS for `T256DynPrimeEngine`, wrapping
//! Hyrax-over-T256 as the underlying F PCS. Implements the paper's
//! IntEval protocol for integer polynomial evaluation at `Z_p` points.
//!
//! Compared to the trivial stub it replaces, this PCS soundly bridges
//! `F_q` arithmetic to `Z_p` evaluations via small-prime fingerprinting:
//! the verifier samples `s` random primes `p_i ≈ 2^{log P}` and opens
//! the F-committed polynomial at `r mod p_i` for each. Because each
//! reduced point is small (`< P`), the F arithmetic stays below `q`
//! and faithfully matches the integer arithmetic, letting the verifier
//! check `to_int(F_y^{(i)}) ≡ int_y (mod p_i)`. By CRT, agreement on
//! `s` independent primes implies the integer evaluation is correct
//! with high probability.

use crate::{
  errors::SpartanError,
  provider::{T256DynPrimeEngine, T256HyraxEngine, pcs::hyrax_pc::HyraxPCS, pt256::t256},
  traits::{
    PrimeFieldExt,
    mod_engine::{ModPCSEngineTrait, SumcheckEngine},
    pcs::PCSEngineTrait,
    transcript::TranscriptReprTrait,
  },
};
use core::marker::PhantomData;
use num_bigint::BigUint;
use serde::{Deserialize, Serialize};

/// Underlying standard PCS: Hyrax over T256.
type Hyrax = HyraxPCS<T256HyraxEngine>;

/// IntEval parameters. Application-level inputs are:
///   - `log T_f`: norm bound on the polynomial `f` being committed.
///   - `log T`:   norm bound on each *limb* of the split polynomial.
///   - `k`:       per-iteration variable count for partial evaluation.
///
/// (Naming matches the paper: `\Bound[f]` and `\Bound` in the LaTeX
/// source render as `T_f` and `T` respectively — see preamble.tex's
/// `\newcommand{\Bound}[1][]{\mathsf{T}_{#1}}`. The `compute_params.py`
/// script uses the same `T` / `log_T` convention.)
///
/// In no-limb-split mode (Phase-3 step B), the polynomial is committed
/// as a single limb, so `T = T_f`. Once limb-splitting lands, `T` is
/// chosen smaller than `T_f` (typically `~32` bits) so each limb fits
/// inside F's characteristic with room for IntEval's intermediate
/// products.
///
/// Module constants: `LAMBDA = 128` (security target), `LOG_Q = 256`
/// (T256's characteristic width). Protocol parameters `(log P, s)` are
/// *derived* from `(log T, k, num_vars)` per the paper's recipe.
///
/// `derive(log_t_f, log_t, k, num_vars)` returns a valid setting;
/// `explicit(...)` lets a caller override `(k, log P, s)` and revalidates.
/// Both go through `validate(num_vars)` which checks the four bounds
/// from §4.4 — Final Eval, Partial Eval Norm, Soundness 1, Soundness 2.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct IntEvalParams {
  /// Per-iteration variables consumed during partial evaluation.
  pub k: usize,
  /// Bit-width upper bound on the random small primes `p_i ∈ [P/2, P]`.
  pub log_p: usize,
  /// Number of random small primes sampled per evaluation.
  pub s: usize,
  /// Norm bound on each *limb* of the split polynomial, in bits.
  /// In no-limb-split mode this equals `log_t_f`.
  pub log_t: usize,
  /// Norm bound on the committed polynomial `f` itself, in bits.
  pub log_t_f: usize,
}

/// Security parameter (bits). The protocol targets `2^{-λ}` soundness.
pub const LAMBDA: usize = 128;

/// Bit-width of the underlying F's characteristic `q`. Fixed at 256 for
/// T256; future engines with other widths would parameterize this.
pub const LOG_Q: usize = 256;

impl IntEvalParams {
  /// Derive a valid `(log P, s)` for the given `(log T_f, log B, k,
  /// num_vars)`. Picks the largest `log P` satisfying Final Eval +
  /// Partial Eval Norm bounds (the latter using `log T`, the limb
  /// bound), then the smallest `s` satisfying Soundness 1.
  pub fn derive(
    log_t_f: usize,
    log_t: usize,
    k: usize,
    num_vars: usize,
  ) -> Result<Self, SpartanError> {
    let log_n = ceil_log2(num_vars.max(1));
    let log_lambda = ceil_log2(LAMBDA);

    // Find max log_p satisfying Partial Evaluation Norm Bound:
    //   k + k·log_p + max(log_t, log_p) < log_q   (uses limb bound T)
    let mut log_p = 0usize;
    for lp in 1..LOG_Q {
      let partial = k + k * lp + log_t.max(lp);
      if partial < LOG_Q {
        log_p = lp;
      } else {
        break;
      }
    }
    if log_p <= 1 {
      return Err(SpartanError::InvalidInputLength {
        reason: format!(
          "IntEvalParams::derive: no log P > 1 satisfies Partial Eval Norm \
           for k={k}, log T={log_t}, log q={LOG_Q}"
        ),
      });
    }

    // Smallest s satisfying Soundness 1: s · (log_p - 5 - log λ - log n) ≥ λ.
    let denom = log_p as isize - 5 - log_lambda as isize - log_n as isize;
    if denom <= 0 {
      return Err(SpartanError::InvalidInputLength {
        reason: format!(
          "IntEvalParams::derive: Soundness 1 denominator non-positive for k={k}, \
           num_vars={num_vars}, derived log_p={log_p}"
        ),
      });
    }
    let s = LAMBDA.div_ceil(denom as usize);

    let p = Self {
      k,
      log_p,
      s,
      log_t,
      log_t_f,
    };
    p.validate(num_vars)?;
    Ok(p)
  }

  /// No-limb-split convenience: derive params with `log T = log T_f`
  /// (single-limb regime, Phase-3 step B).
  pub fn derive_no_limb_split(
    log_t_f: usize,
    k: usize,
    num_vars: usize,
  ) -> Result<Self, SpartanError> {
    Self::derive(log_t_f, log_t_f, k, num_vars)
  }

  /// Use explicit `(k, log P, s, log T, log T_f)`. Validates against
  /// `num_vars` so a caller-tuned configuration can't bypass the bound
  /// checks. Errors if any of the four bounds is violated.
  pub fn explicit(
    k: usize,
    log_p: usize,
    s: usize,
    log_t: usize,
    log_t_f: usize,
    num_vars: usize,
  ) -> Result<Self, SpartanError> {
    let p = Self {
      k,
      log_p,
      s,
      log_t,
      log_t_f,
    };
    p.validate(num_vars)?;
    Ok(p)
  }

  /// Check all four bounds from §4.4. Each is evaluated in log-space to
  /// avoid overflow; the comparisons match the paper's inequalities
  /// after taking `log_2` of both sides.
  pub fn validate(&self, num_vars: usize) -> Result<(), SpartanError> {
    let log_n = ceil_log2(num_vars.max(1));
    let log_lambda = ceil_log2(LAMBDA);

    // Final Evaluation Bound: 2^k * P^(k+1) < q
    //   log: k + (k+1)·log_p < log_q
    let final_eval_lhs = self.k + (self.k + 1) * self.log_p;
    if final_eval_lhs >= LOG_Q {
      return Err(SpartanError::InvalidInputLength {
        reason: format!(
          "IntEval Final Evaluation Bound violated: k + (k+1)·log_p = {} >= log_q = {}",
          final_eval_lhs, LOG_Q
        ),
      });
    }

    // Partial Evaluation Norm Bound: 2^k · P^k · max(T, P) <= (q-P)/2
    //   log (approximate, dropping the -P-1 below q): k + k·log_p + max(log_t, log_p) < log_q
    // Uses `log_t` (the *limb* bound), not `log_t_f`, since IntEval
    // operates on the (possibly limb-split) polynomial.
    let partial_norm_lhs = self.k + self.k * self.log_p + self.log_t.max(self.log_p);
    if partial_norm_lhs >= LOG_Q {
      return Err(SpartanError::InvalidInputLength {
        reason: format!(
          "IntEval Partial Evaluation Norm Bound violated: k + k·log_p + max(log_B, log_p) = {} >= log_q = {}",
          partial_norm_lhs, LOG_Q
        ),
      });
    }

    // Sanity: `log_t > log_t_f` doesn't make sense — the limb bound
    // can't exceed the polynomial bound.
    if self.log_t > self.log_t_f {
      return Err(SpartanError::InvalidInputLength {
        reason: format!(
          "IntEvalParams: log_t ({}) must not exceed log_t_f ({})",
          self.log_t, self.log_t_f
        ),
      });
    }

    // Soundness Bound 1: (32 λ n / P)^s <= 2^{-λ}
    //   log: s · (5 + log λ + log n - log_p) <= -λ
    //   <=>  s · (log_p - 5 - log λ - log n) >= λ
    let log_inner = self.log_p as isize - 5 - log_lambda as isize - log_n as isize;
    if log_inner <= 0 || (self.s as isize) * log_inner < (LAMBDA as isize) {
      return Err(SpartanError::InvalidInputLength {
        reason: format!(
          "IntEval Soundness Bound 1 violated: s·(log_p - 5 - log λ - log n) = {} < λ = {}",
          (self.s as isize) * log_inner,
          LAMBDA
        ),
      });
    }

    // Soundness Bound 2: s · n / |F| <= 2^{-λ}
    //   log: log(s·n) - log_q <= -λ
    //   <=>  log_q >= λ + log(s·n)
    let log_sn = ceil_log2((self.s * num_vars).max(1));
    if LOG_Q < LAMBDA + log_sn {
      return Err(SpartanError::InvalidInputLength {
        reason: format!(
          "IntEval Soundness Bound 2 violated: log_q = {} < λ + log(s·n) = {}",
          LOG_Q,
          LAMBDA + log_sn
        ),
      });
    }

    Ok(())
  }
}

/// Ceiling `log_2`. `ceil_log2(0)` returns 0 (callers guard with `.max(1)`).
fn ceil_log2(x: usize) -> usize {
  if x <= 1 {
    return 0;
  }
  (usize::BITS - (x - 1).leading_zeros()) as usize
}

/// Mod-PCS commitment key wraps Hyrax's plus the IntEval parameters.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct IntEvalCommitmentKey {
  pub(crate) inner: <Hyrax as PCSEngineTrait<T256HyraxEngine>>::CommitmentKey,
  pub(crate) params: IntEvalParams,
}

/// Verifier key wraps Hyrax's plus the IntEval parameters.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct IntEvalVerifierKey {
  pub(crate) inner: <Hyrax as PCSEngineTrait<T256HyraxEngine>>::VerifierKey,
  pub(crate) params: IntEvalParams,
}

/// Commitment is just the underlying Hyrax commitment to the F-cast
/// polynomial. The IntEval protocol runs entirely at eval time.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct IntEvalCommitment {
  pub(crate) inner: <Hyrax as PCSEngineTrait<T256HyraxEngine>>::Commitment,
}

impl TranscriptReprTrait for IntEvalCommitment {
  fn to_transcript_bytes(&self) -> Vec<u8> {
    self.inner.to_transcript_bytes()
  }
}

/// Blind delegates to Hyrax's.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct IntEvalBlind {
  pub(crate) inner: <Hyrax as PCSEngineTrait<T256HyraxEngine>>::Blind,
}

/// Evaluation argument — currently a placeholder containing only the
/// underlying Hyrax eval-argument. Step B will extend this to carry the
/// small-prime opens; step C will carry the partial-eval oracles.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct IntEvalEvalArg {
  pub(crate) _todo: (),
}

/// `BigUint → t256::Scalar` via 64-byte wide reduction. Value-preserving
/// for inputs below the scalar field, otherwise reduces uniformly. Same
/// convention as `BridgeModPCS`. Phase 3 step D will add the range check
/// that turns this into a *sound* commitment to a bounded integer.
fn biguint_to_scalar(v: &BigUint) -> t256::Scalar {
  let mut bytes = v.to_bytes_le();
  bytes.resize(64, 0);
  <t256::Scalar as PrimeFieldExt>::from_uniform(&bytes)
}

/// Sound Mod-PCS for `T256DynPrimeEngine`. See module docs.
#[derive(Clone)]
pub struct IntEvalModPCS {
  _phantom: PhantomData<()>,
}

/// Application-level defaults used by the trait `setup` when the caller
/// doesn't pass explicit `IntEvalParams`. These are the application-
/// level bounds and iteration knob; the rest of the protocol parameters
/// (`log P`, `s`) are derived from them. Use `setup_with_params` to
/// override.
///
/// Default polynomial norm bound used by trait `setup` (`log_2(T_f)`).
pub const DEFAULT_LOG_T_F: usize = 32;
/// Default per-iteration variable count used by trait `setup`. Matches
/// the paper's recommended `k = ⌈log λ⌉`.
pub const DEFAULT_K: usize = 7;

impl IntEvalModPCS {
  /// Explicit-params setup. Validates the params against `num_vars =
  /// log_2(n)` so caller-supplied configurations can't bypass the
  /// IntEval soundness bounds.
  pub fn setup_with_params(
    label: &'static [u8],
    n: usize,
    width: usize,
    params: IntEvalParams,
  ) -> Result<
    (
      <Self as ModPCSEngineTrait<T256DynPrimeEngine>>::CommitmentKey,
      <Self as ModPCSEngineTrait<T256DynPrimeEngine>>::VerifierKey,
    ),
    SpartanError,
  > {
    let num_vars = ceil_log2(n.max(1));
    params.validate(num_vars)?;
    let (inner_ck, inner_vk) = Hyrax::setup(label, n, width);
    Ok((
      IntEvalCommitmentKey {
        inner: inner_ck,
        params: params.clone(),
      },
      IntEvalVerifierKey {
        inner: inner_vk,
        params,
      },
    ))
  }
}

impl ModPCSEngineTrait<T256DynPrimeEngine> for IntEvalModPCS {
  type CommitmentKey = IntEvalCommitmentKey;
  type VerifierKey = IntEvalVerifierKey;
  type Commitment = IntEvalCommitment;
  type Blind = IntEvalBlind;
  type EvaluationArgument = IntEvalEvalArg;

  /// Trait-driven setup: derive `IntEvalParams` from the application
  /// defaults `(DEFAULT_LAMBDA, DEFAULT_LOG_T_F)` and the polynomial
  /// size. Panics if the derivation fails (which only happens for
  /// pathologically small `n`); callers that need control over the
  /// security or norm-bound parameters should use `setup_with_params`.
  fn setup(
    label: &'static [u8],
    n: usize,
    width: usize,
  ) -> (Self::CommitmentKey, Self::VerifierKey) {
    let num_vars = ceil_log2(n.max(1));
    let params = IntEvalParams::derive_no_limb_split(DEFAULT_LOG_T_F, DEFAULT_K, num_vars).expect(
      "default IntEvalParams derivation must satisfy the paper's bounds; \
         override with `setup_with_params` to use tighter parameters",
    );
    let (inner_ck, inner_vk) = Hyrax::setup(label, n, width);
    (
      IntEvalCommitmentKey {
        inner: inner_ck,
        params: params.clone(),
      },
      IntEvalVerifierKey {
        inner: inner_vk,
        params,
      },
    )
  }

  fn precompute_ck(ck: &Self::CommitmentKey) {
    Hyrax::precompute_ck(&ck.inner)
  }

  fn blind(ck: &Self::CommitmentKey, n: usize) -> Self::Blind {
    IntEvalBlind {
      inner: Hyrax::blind(&ck.inner, n),
    }
  }

  fn commit(
    ck: &Self::CommitmentKey,
    v: &[BigUint],
    r: &Self::Blind,
    is_small: bool,
  ) -> Result<Self::Commitment, SpartanError> {
    // TODO Phase 3 step D: range-check |v[i]| < B_f before committing.
    // For now, cast directly. Soundness relies on the bound being met.
    let v_fq: Vec<t256::Scalar> = v.iter().map(biguint_to_scalar).collect();
    let inner = Hyrax::commit(&ck.inner, &v_fq, &r.inner, is_small)?;
    Ok(IntEvalCommitment { inner })
  }

  fn check_commitment(comm: &Self::Commitment, n: usize, width: usize) -> Result<(), SpartanError> {
    Hyrax::check_commitment(&comm.inner, n, width)
  }

  fn prove(
    _ck: &Self::CommitmentKey,
    _ck_eval: &Self::CommitmentKey,
    _transcript: &mut <T256DynPrimeEngine as SumcheckEngine>::TE,
    _comm: &Self::Commitment,
    _poly: &[BigUint],
    _blind: &Self::Blind,
    _point: &[<T256DynPrimeEngine as SumcheckEngine>::Scalar],
    _comm_eval: &Self::Commitment,
    _blind_eval: &Self::Blind,
  ) -> Result<Self::EvaluationArgument, SpartanError> {
    // TODO Phase 3 step B: sample s small primes from the transcript,
    // open the F-commitment at `point mod p_i` for each, package the
    // s eval claims into the evaluation argument.
    Err(SpartanError::InternalError {
      reason: "IntEvalModPCS::prove not yet implemented (Phase 3 step B)".to_string(),
    })
  }

  fn verify(
    _vk: &Self::VerifierKey,
    _ck_eval: &Self::CommitmentKey,
    _transcript: &mut <T256DynPrimeEngine as SumcheckEngine>::TE,
    _comm: &Self::Commitment,
    _point: &[<T256DynPrimeEngine as SumcheckEngine>::Scalar],
    _comm_eval: &Self::Commitment,
    _arg: &Self::EvaluationArgument,
  ) -> Result<(), SpartanError> {
    // TODO Phase 3 step B: re-sample the same s small primes from the
    // transcript, verify each F-PCS opening, check `to_int(F_y^(i)) ≡
    // int_v' mod p_i` and the CRT-implied congruence with the claimed
    // Z_p eval.
    Err(SpartanError::InternalError {
      reason: "IntEvalModPCS::verify not yet implemented (Phase 3 step B)".to_string(),
    })
  }
}

#[cfg(test)]
mod tests {
  use super::*;
  use crate::dyn_prime::DynPrime;
  use crate::traits::mod_engine::SumcheckField;
  use crate::traits::transcript::TranscriptEngineTrait;

  type ME = T256DynPrimeEngine;
  type MP = IntEvalModPCS;
  type DP = DynPrime<4>;

  /// Setup + commit round-trip: an IntEval-committed polynomial commits
  /// to the same Hyrax handle as a direct Hyrax commit of the cast
  /// F-valued polynomial. Sanity check that the wrapper isn't mangling
  /// the underlying commitment.
  #[test]
  fn commit_delegates_to_hyrax() {
    let n = 16usize;
    let (ck, _vk) = <MP as ModPCSEngineTrait<ME>>::setup(b"inteval-test", n, 256);
    let poly: Vec<BigUint> = (0..n).map(|i| BigUint::from(7u32 * i as u32 + 3)).collect();
    let blind = <MP as ModPCSEngineTrait<ME>>::blind(&ck, n);
    let comm = <MP as ModPCSEngineTrait<ME>>::commit(&ck, &poly, &blind, false).unwrap();

    // Re-commit directly via Hyrax and confirm equality.
    let poly_fq: Vec<t256::Scalar> = poly.iter().map(biguint_to_scalar).collect();
    let direct = Hyrax::commit(&ck.inner, &poly_fq, &blind.inner, false).unwrap();
    assert_eq!(comm.inner, direct);
  }

  /// `derive` produces params that pass `validate` for the variable
  /// counts our test SNARKs use.
  #[test]
  fn derive_default_params_valid() {
    for num_vars in [1usize, 2, 4, 8, 16, 25] {
      let p = IntEvalParams::derive_no_limb_split(DEFAULT_LOG_T_F, DEFAULT_K, num_vars).unwrap();
      p.validate(num_vars).unwrap();
      assert!(p.k >= 1 && p.k <= 12);
      assert!(p.log_p > 5 + ceil_log2(LAMBDA) + ceil_log2(num_vars.max(1)));
      assert!(p.s >= 1);
    }
  }

  /// Print derived params for the variable counts our tests use, as a
  /// human-readable record. Failing the test isn't the goal; the
  /// printed values document what `derive` actually picks.
  #[test]
  fn derive_picks_reasonable_params() {
    for num_vars in [4usize, 8, 16, 25] {
      let p = IntEvalParams::derive_no_limb_split(DEFAULT_LOG_T_F, DEFAULT_K, num_vars).unwrap();
      eprintln!(
        "derive(log_T_f={}, k={}, n={}) → log_P={}, s={}",
        p.log_t_f, p.k, num_vars, p.log_p, p.s
      );
    }
  }

  /// `explicit` rejects a config whose Soundness Bound 1 fails: a small
  /// `log_p` paired with a small `s` gives a too-large soundness error.
  #[test]
  fn explicit_rejects_bad_soundness() {
    let err = IntEvalParams::explicit(
      /* k */ 7, /* log_p */ 12, // way too small: soundness_1 = (32·128·n/2^12)^s
      /* s */ 1, /* log_t */ 32, /* log_t_f */ 32, /* num_vars */ 8,
    )
    .unwrap_err();
    assert!(matches!(err, SpartanError::InvalidInputLength { .. }));
  }

  /// `explicit` rejects a config whose Partial Evaluation Norm Bound
  /// fails: large `k` × large `log_p` overflows the field.
  #[test]
  fn explicit_rejects_partial_norm_overflow() {
    let err = IntEvalParams::explicit(
      /* k */ 12, /* log_p */ 40, /* s */ 5, /* log_t */ 32,
      /* log_t_f */ 32, /* num_vars */ 8,
    )
    .unwrap_err();
    assert!(matches!(err, SpartanError::InvalidInputLength { .. }));
  }

  /// `setup_with_params` accepts a valid override, rejects a bad one.
  #[test]
  fn setup_with_params_round_trips_overrides() {
    let n = 16usize;
    let p = IntEvalParams::derive_no_limb_split(DEFAULT_LOG_T_F, DEFAULT_K, ceil_log2(n)).unwrap();
    let (_ck, _vk) = IntEvalModPCS::setup_with_params(b"override", n, 256, p).unwrap();

    // Bad params: zero `s` makes soundness_1 fail trivially.
    let bad = IntEvalParams {
      k: 7,
      log_p: 20,
      s: 0,
      log_t: 32,
      log_t_f: 32,
    };
    let err = IntEvalModPCS::setup_with_params(b"override", n, 256, bad).unwrap_err();
    assert!(matches!(err, SpartanError::InvalidInputLength { .. }));
  }

  /// prove/verify currently error with a not-yet-implemented marker —
  /// pins that the trait surface compiles and that we land step B's
  /// implementation here.
  #[test]
  fn prove_verify_todo_markers() {
    let n = 4usize;
    let (ck, vk) = <MP as ModPCSEngineTrait<ME>>::setup(b"inteval-test", n, 256);
    let (ck_eval, _) = <MP as ModPCSEngineTrait<ME>>::setup(b"ck_eval", 1, 1);

    let params = crate::provider::pcs::bridge_modpcs::t256_scalar_params();
    let poly: Vec<BigUint> = (0..n).map(|i| BigUint::from(i as u32 + 1)).collect();
    let point: Vec<DP> = (0..2)
      .map(|i| DP::from_u64(&params, (i as u64) + 1))
      .collect();
    let blind = <MP as ModPCSEngineTrait<ME>>::blind(&ck, n);
    let comm = <MP as ModPCSEngineTrait<ME>>::commit(&ck, &poly, &blind, false).unwrap();
    let blind_eval = <MP as ModPCSEngineTrait<ME>>::blind(&ck_eval, 1);
    let comm_eval =
      <MP as ModPCSEngineTrait<ME>>::commit(&ck_eval, &[BigUint::from(0u32)], &blind_eval, false)
        .unwrap();

    let mut pt = <ME as SumcheckEngine>::TE::new_with_params(b"smoke", params);
    let err = <MP as ModPCSEngineTrait<ME>>::prove(
      &ck,
      &ck_eval,
      &mut pt,
      &comm,
      &poly,
      &blind,
      &point,
      &comm_eval,
      &blind_eval,
    )
    .unwrap_err();
    assert!(matches!(err, SpartanError::InternalError { .. }));

    let arg = IntEvalEvalArg { _todo: () };
    let mut vt = <ME as SumcheckEngine>::TE::new_with_params(b"smoke", params);
    let err = <MP as ModPCSEngineTrait<ME>>::verify(
      &vk, &ck_eval, &mut vt, &comm, &point, &comm_eval, &arg,
    )
    .unwrap_err();
    assert!(matches!(err, SpartanError::InternalError { .. }));
  }
}
