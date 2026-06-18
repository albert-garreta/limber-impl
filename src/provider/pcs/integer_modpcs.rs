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

//! `IntegerModPCS`: sound Mod-PCS for `T256DynPrimeEngine`, wrapping
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
  polys::eq::EqPolynomial,
  provider::{
    T256DynPrimeEngine, T256HyraxEngine,
    keccak::Keccak256Transcript,
    pcs::hyrax_pc::{HyraxBlind, HyraxPCS},
    pt256::t256,
  },
  start_span,
  traits::{
    PrimeFieldExt,
    mod_engine::{ModPCSEngineTrait, SumcheckEngine, SumcheckField},
    pcs::PCSEngineTrait,
    transcript::{ByteTranscript, TranscriptEngineTrait, TranscriptReprTrait},
  },
};
use core::marker::PhantomData;
use ff::{Field, PrimeField};
use num_bigint::{BigInt, BigUint, Sign};
use num_integer::Integer;
use num_traits::{One, Zero};
use rayon::prelude::*;
use serde::{Deserialize, Serialize};
use tracing::info;

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
  /// Number of limbs per polynomial coefficient: `⌈log_t_f / log_t⌉`.
  /// Setup-fixed and public — both prover and verifier read this from
  /// the params they share. `1` in no-limb-split mode.
  pub numlimb: usize,
  /// Bit-width of the limb index, `⌈log_2 numlimb⌉`. `0` when
  /// `numlimb = 1` (no extra polynomial variables needed).
  pub numlimb_var: usize,
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
    let nl_pre = numlimb(log_t_f, log_t);
    let nlv_pre = numlimb_var(nl_pre);
    let num_vars_total = num_vars + nlv_pre;

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

    // Smallest s satisfying the prime-divisibility soundness bound
    //   (log_P(y) / (π(P) − π(P/2)))^s ≤ 2^{−λ},
    // where log2(y) = n + λ·n + log_t bounds the integer difference between a
    // false and the true partial evaluation, and log_P(y) upper-bounds how
    // many primes ≥ P/2 can divide it. `bits_per_prime` is the soundness each
    // random small prime in (P/2, P] contributes; the prime count π(P)−π(P/2)
    // is lower-bounded (Dusart/Rosser–Schoenfeld) so s stays sound. Replaces
    // the older crude `(32 λ n / P)` union bound, which over-provisioned s.
    let bits_per_prime = soundness_bits_per_prime(log_p, num_vars_total, log_t);
    if bits_per_prime <= 0.0 {
      return Err(SpartanError::InvalidInputLength {
        reason: format!(
          "IntEvalParams::derive: prime-divisibility soundness gives ≤ 0 bits per \
           prime for k={k}, num_vars={num_vars}, derived log_p={log_p}"
        ),
      });
    }
    let s = (LAMBDA as f64 / bits_per_prime).ceil() as usize;

    let nl = numlimb(log_t_f, log_t);
    let p = Self {
      k,
      log_p,
      s,
      log_t,
      log_t_f,
      numlimb: nl,
      numlimb_var: numlimb_var(nl),
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
    let nl = numlimb(log_t_f, log_t);
    let p = Self {
      k,
      log_p,
      s,
      log_t,
      log_t_f,
      numlimb: nl,
      numlimb_var: numlimb_var(nl),
    };
    p.validate(num_vars)?;
    Ok(p)
  }

  /// Check all four bounds from §4.4. Each is evaluated in log-space to
  /// avoid overflow; the comparisons match the paper's inequalities
  /// after taking `log_2` of both sides. `num_vars` is the *original*
  /// polynomial variable count — the limb-split polynomial has
  /// `num_vars + numlimb_var` variables, and that's what enters the
  /// soundness bounds.
  pub fn validate(&self, num_vars: usize) -> Result<(), SpartanError> {
    let num_vars_total = num_vars + self.numlimb_var;

    // Limb-decomposition self-consistency: `numlimb` and `numlimb_var`
    // must match the formulas implied by `(log_t, log_t_f)`. Catches
    // hand-rolled `IntEvalParams { ... }` literals that get the
    // relation wrong.
    let expected_nl = numlimb(self.log_t_f, self.log_t);
    if self.numlimb != expected_nl {
      return Err(SpartanError::InvalidInputLength {
        reason: format!(
          "IntEvalParams: numlimb = {} does not match ⌈log_T_f / log_T⌉ = ⌈{}/{}⌉ = {}",
          self.numlimb, self.log_t_f, self.log_t, expected_nl
        ),
      });
    }
    let expected_nlv = numlimb_var(self.numlimb);
    if self.numlimb_var != expected_nlv {
      return Err(SpartanError::InvalidInputLength {
        reason: format!(
          "IntEvalParams: numlimb_var = {} does not match ⌈log_2 numlimb⌉ = ⌈log_2 {}⌉ = {}",
          self.numlimb_var, self.numlimb, expected_nlv
        ),
      });
    }

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

    // Soundness Bound 1 (prime divisibility): (log_P(y) / (π(P) − π(P/2)))^s ≤ 2^{−λ}
    //   <=>  s · bits_per_prime ≥ λ,  bits_per_prime = log2(π(P)−π(P/2)) − log2(log_P y).
    let bits_per_prime = soundness_bits_per_prime(self.log_p, num_vars_total, self.log_t);
    if bits_per_prime <= 0.0 || (self.s as f64) * bits_per_prime < LAMBDA as f64 {
      return Err(SpartanError::InvalidInputLength {
        reason: format!(
          "IntEval Soundness Bound 1 violated: s·bits_per_prime = {:.2} < λ = {}",
          (self.s as f64) * bits_per_prime,
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

  /// Coarse prover-cost estimate for these params on a `num_vars`-variable
  /// polynomial, in "committed scalar" units (one unit ≈ the amortized
  /// per-scalar cost of a Hyrax commit plus its share of the batched
  /// open). This is a *ranking heuristic* for `derive_optimized`, not a
  /// performance guarantee — soundness is enforced independently by
  /// `validate`, so a misranked candidate costs time, never security.
  ///
  /// Terms, mirroring the prove path:
  ///   - `f_limb` commit/open plus its `T`-bound 16-bit-chunk range
  ///     checks: `2^{n_tot} · (2 + ⌈log T / 16⌉)`;
  ///   - per iteration layer `j` (`m_j = 2^{n_tot − j·k}` slots): the
  ///     stacked `(a_j, b_j)` commit/open (`2 · s_pad · m_j`), the LogUp
  ///     chunk commitments for the shifted `a_j` (width `log P + 1`) and
  ///     `b_j` (width `log q − log P + 1`), and the integer
  ///     partial-evaluation work (`s` chains × `m_j · 2^k` bigint
  ///     mult-adds of `⌈(k·(log P + 1) + log T)/64⌉`-word operands);
  ///   - eq-tensor weight assembly for the `s` per-chain claims of the
  ///     final batched opens: `s · 2^{n_tot}` field ops.
  pub fn estimated_prover_cost(&self, num_vars: usize) -> f64 {
    /// One 256-bit field op, in committed-scalar units.
    const FIELD_OP: f64 = 0.05;
    /// One 64-bit bigint word op, in committed-scalar units.
    const BIGINT_WORD: f64 = 0.02;

    let two = |e: usize| (e as f64).exp2();
    let n_tot = num_vars + self.numlimb_var;
    let s = self.s as f64;
    let s_pad = self.s.next_power_of_two() as f64;

    // f_limb commit/open + T-bound chunk range checks.
    let f_chunks = self.log_t.div_ceil(CHUNK_BITS) as f64;
    let mut cost = two(n_tot) * (2.0 + f_chunks);

    // Iteration layers (none when n_tot <= k).
    let t_layers = n_tot.saturating_sub(self.k).div_ceil(self.k);
    let ab_chunks = ((self.log_p + 1).div_ceil(CHUNK_BITS)
      + (LOG_Q - self.log_p + 1).div_ceil(CHUNK_BITS)) as f64;
    let words = (self.k * (self.log_p + 1) + self.log_t).div_ceil(64) as f64;
    for j in 1..=t_layers {
      let m = two(n_tot - j * self.k);
      // Stacked commit/open + LogUp chunk commitments.
      cost += m * (2.0 * s_pad + s * ab_chunks);
      // Integer partial evaluation: input size m · 2^k per chain.
      cost += BIGINT_WORD * s * m * two(self.k) * words;
    }

    // Per-claim eq-tensor weights in the batched opens.
    cost += FIELD_OP * s * two(n_tot);
    cost
  }

  /// Derive params *optimized for the given input length*: search over
  /// the per-iteration variable count `k` and the limb bound `log T`,
  /// deriving the dependent `(log P, s)` for each candidate via
  /// [`Self::derive`], and return the candidate minimizing
  /// [`Self::estimated_prover_cost`] (ties broken by smaller `s`, i.e.
  /// cheaper verifier and smaller proofs, then smaller `numlimb`).
  ///
  /// The search space is tiny — `k ∈ [1, num_vars + numlimb_var]` per
  /// limb candidate, and limb candidates halve `log T` from `log T_f`
  /// down to the 16-bit range-check chunk width — so this is cheap
  /// enough to run at every setup. Every candidate passes through
  /// `derive`'s `validate`, so the optimizer can only affect
  /// performance, never soundness.
  pub fn derive_optimized(log_t_f: usize, num_vars: usize) -> Result<Self, SpartanError> {
    // Limb candidates: numlimb = 2^v, log T = ⌈log T_f / 2^v⌉. Splitting
    // below one range-check chunk (16 bits) cannot reduce the per-value
    // chunk count further, so stop there.
    let mut log_t_candidates: Vec<usize> = Vec::new();
    for v in 0..=24usize {
      let log_t = log_t_f.div_ceil(1usize << v).max(1);
      if log_t_candidates.last() != Some(&log_t) {
        log_t_candidates.push(log_t);
      }
      if log_t <= CHUNK_BITS {
        break;
      }
    }

    let mut best: Option<(f64, Self)> = None;
    let mut last_err: Option<SpartanError> = None;
    for &log_t in &log_t_candidates {
      let nlv = numlimb_var(numlimb(log_t_f, log_t));
      // k > n_tot behaves like k = n_tot (zero iterations) but only
      // tightens the norm bounds, so cap the search there.
      let k_max = (num_vars + nlv).max(1);
      for k in 1..=k_max {
        match Self::derive(log_t_f, log_t, k, num_vars) {
          Ok(p) => {
            let cost = p.estimated_prover_cost(num_vars);
            let better = match &best {
              None => true,
              Some((bc, bp)) => {
                cost < *bc || (cost == *bc && (p.s, p.numlimb) < (bp.s, bp.numlimb))
              }
            };
            if better {
              best = Some((cost, p));
            }
          }
          Err(e) => last_err = Some(e),
        }
      }
    }

    best.map(|(_, p)| p).ok_or_else(|| {
      last_err.unwrap_or(SpartanError::InvalidInputLength {
        reason: format!(
          "IntEvalParams::derive_optimized: no valid (k, log T) candidate for \
           log_t_f={log_t_f}, num_vars={num_vars}"
        ),
      })
    })
  }
}

/// Ceiling `log_2`. `ceil_log2(0)` returns 0 (callers guard with `.max(1)`).
fn ceil_log2(x: usize) -> usize {
  if x <= 1 {
    return 0;
  }
  (usize::BITS - (x - 1).leading_zeros()) as usize
}

/// Strict lower bound on `π(2^log2_x)` (the prime-counting function). Uses
/// Dusart's (2010) `π(x) ≥ (x/ln x)(1 + 1/ln x)` for `x ≥ 599`, and the
/// Rosser–Schoenfeld `π(x) > x/ln x` (valid `x ≥ 17`) below that. Returns a
/// lower bound so downstream prime-count soundness estimates stay conservative.
fn pi_lower_2pow(log2_x: usize) -> f64 {
  let x = (log2_x as f64).exp2();
  let lnx = (log2_x as f64) * core::f64::consts::LN_2;
  if x >= 599.0 {
    (x / lnx) * (1.0 + 1.0 / lnx)
  } else {
    x / lnx
  }
}

/// Upper bound on `π(2^log2_x)` via Dusart's `π(x) ≤ (x/ln x)(1 + 1.2762/ln x)`
/// (valid `x ≥ 2`).
fn pi_upper_2pow(log2_x: usize) -> f64 {
  let x = (log2_x as f64).exp2();
  let lnx = (log2_x as f64) * core::f64::consts::LN_2;
  (x / lnx) * (1.0 + 1.2762 / lnx)
}

/// `log2` of a lower bound on the number of primes in `(P/2, P]`, `P = 2^log_p`.
fn log2_primes_in_top_half(log_p: usize) -> f64 {
  let count = (pi_lower_2pow(log_p) - pi_upper_2pow(log_p.saturating_sub(1))).max(1.0);
  count.log2()
}

/// Soundness (in bits) each random small prime `p ∈ (P/2, P]`, `P = 2^log_p`,
/// contributes to the IntEval CRT fingerprint:
///   `log2(π(P) − π(P/2)) − log2(log_P(y))`,
/// with `log2(y) = n + λ·n + log_t` the bound on the integer difference between
/// a false and the true partial evaluation, and `log_P(y)` an upper bound on
/// how many primes `≥ P/2` can divide it. `n` is the limb-split polynomial's
/// variable count. A larger value ⇒ fewer primes `s` needed. Primes below
/// `2^5` are too sparse for the bounds, so they return a rejecting value.
fn soundness_bits_per_prime(log_p: usize, n: usize, log_t: usize) -> f64 {
  if log_p < 5 {
    return -1.0;
  }
  let log2_y = (n as f64) * (1.0 + LAMBDA as f64) + (log_t as f64);
  let log_p_y = (log2_y / (log_p as f64)).max(1.0);
  log2_primes_in_top_half(log_p) - log_p_y.log2()
}

/// Mod-PCS commitment key wraps Hyrax's plus the IntEval parameters.
///
/// `eval` is a size-1 Hyrax key the IntEval protocol uses internally to
/// form the per-opening eval commitment `G^{f_y}` (which the verifier
/// reconstructs locally). It lives inside the Mod-PCS key — not on the
/// universal `ModPCSEngineTrait` surface — so the trait stays
/// PCS-agnostic (a hash/FRI Mod-PCS would carry no such key).
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct IntegerModCommitmentKey {
  pub(crate) inner: <Hyrax as PCSEngineTrait<T256HyraxEngine>>::CommitmentKey,
  pub(crate) eval: <Hyrax as PCSEngineTrait<T256HyraxEngine>>::CommitmentKey,
  pub(crate) params: IntEvalParams,
}

/// Verifier key wraps Hyrax's plus the IntEval parameters.
///
/// `eval` mirrors the commitment key's size-1 eval key: the verifier
/// reconstructs the eval commitment `G^{f_y}` locally during opening
/// verification, so it needs the same generators. Kept inside the key
/// rather than on the trait surface (PCS-agnostic).
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct IntegerModVerifierKey {
  pub(crate) inner: <Hyrax as PCSEngineTrait<T256HyraxEngine>>::VerifierKey,
  pub(crate) eval: <Hyrax as PCSEngineTrait<T256HyraxEngine>>::CommitmentKey,
  pub(crate) params: IntEvalParams,
}

/// Commitment is just the underlying Hyrax commitment to the F-cast
/// polynomial. The IntEval protocol runs entirely at eval time.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct IntegerModCommitment {
  pub(crate) inner: <Hyrax as PCSEngineTrait<T256HyraxEngine>>::Commitment,
}

impl TranscriptReprTrait for IntegerModCommitment {
  fn to_transcript_bytes(&self) -> Vec<u8> {
    self.inner.to_transcript_bytes()
  }
}

/// Blind delegates to Hyrax's.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct IntegerModBlind {
  pub(crate) inner: <Hyrax as PCSEngineTrait<T256HyraxEngine>>::Blind,
}

/// One per-small-prime opening: the F-side evaluation `F_y^(i)` and the
/// Hyrax evaluation argument. The eval commitment `comm_eval = G^{f_y}`
/// is deterministic in `f_y` (zero-blind), so the verifier reconstructs
/// it locally — no `blind_eval` shipped.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct SmallPrimeOpening {
  /// F_y^(i) = f_F(r mod p_i), the F-evaluation at the small-prime-
  /// reduced point. Sent in the clear; verifier checks it for the
  /// CRT congruence `to_int(F_y^(i)) ≡ int_v' (mod p_i)`.
  pub f_y: t256::Scalar,
  /// Hyrax evaluation argument for the opening at `r mod p_i`.
  pub hyrax_arg: <Hyrax as PCSEngineTrait<T256HyraxEngine>>::EvaluationArgument,
}

/// One iteration's identity-check evaluation claims. The commitments
/// live in the per-layer stacked `(role, chain, x)` polynomials
/// ([`IntEvalArgument::ab_comms`]); every claim here is discharged by the
/// final batched opens.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct IterationOracles {
  /// Claimed `a_{j-1}(γ_ext)` where `γ_ext = (γ[0..n-jk], r^(i)[n-jk..n-(j-1)k])`.
  /// A claim on the input commitment for `j=1`, else on layer `j-1`'s
  /// stacked commitment at the chain's boolean-extended point.
  pub a_prev_eval: t256::Scalar,
  /// Claimed `a_j(γ[0..n-jk])` — a claim on layer `j`'s stacked
  /// commitment at `(0, bits(chain), γ_prefix)`.
  pub a_curr_eval: t256::Scalar,
  /// Claimed `b_j(γ[0..n-jk])` — same point with role bit 1.
  pub b_curr_eval: t256::Scalar,
}

/// Per-prime chain: `t = ⌈(n-k)/k⌉` iterations plus the claimed
/// final-remainder evaluation `a_t(r^(i)[0..n-tk])` (a claim on the input
/// commitment for `t = 0`, else on the last stacked layer).
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ChainData {
  /// Per-iteration identity-check evaluation claims.
  pub iterations: Vec<IterationOracles>,
  /// Claimed `a_t(r^(i)[0..n-tk])`, used by the CRT check.
  pub final_eval: t256::Scalar,
}

/// Evaluation argument: the prover-sent integer evaluation `int_v'`,
/// the reduction-sumcheck round polynomials (Phase-3 step D3), and one
/// per-prime chain.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct IntEvalArgument {
  /// Per-round compressed univariate polynomial of the reduction
  /// sumcheck `sum_k limb(k) · f_limb(int_r, k) ≡_p int_y`. Each inner
  /// vector is the round poly's coefficients excluding the linear term,
  /// stored as `BigUint` (canonical representatives mod `p`) so the
  /// `IntEvalArgument` stays Serde-friendly without dragging
  /// `SumcheckProof<T256DynPrimeEngine>` through serde plumbing.
  /// `numlimb_var` entries total; empty when `numlimb_var = 0`
  /// (no-limb-split mode, the reduction sumcheck is degenerate).
  pub reduction_round_polys: Vec<Vec<BigUint>>,
  /// `int_v' = f_limb(int_r, int_r_k)` as a signed integer. Negative
  /// values come from `(1 - r_i)` factors in the multilinear chi. For
  /// `numlimb_var = 0` this equals the integer evaluation of `f` at
  /// `int_r`.
  pub int_v_prime: BigInt,
  /// One per small prime sampled from the transcript. Length matches
  /// `params.s`.
  pub chains: Vec<ChainData>,
  /// One stacked commitment per iteration layer `j ∈ [1, t]`, holding ALL
  /// chains' shifted `a_j` and `b_j`: index `(role, chain, x)` with role
  /// `0 = a`, `1 = b` and chains padded to the next power of two. Empty
  /// when `t = 0`.
  pub(crate) ab_comms: Vec<<Hyrax as PCSEngineTrait<T256HyraxEngine>>::Commitment>,
  /// ONE shared LogUp-GKR range check covering all `(bound, size)`
  /// batch groups. Canonical batch order is `f_limb`, then for each
  /// iteration `j = 1..=t` the `a_j` batch (all `s` chains) and the
  /// `b_j` batch — `1 + 2t` batches (just `f_limb` when `t = 0`), each
  /// with its own 16-bit-chunk commitment and reconstruction sumcheck,
  /// all sharing one multiplicity table and one table-side GKR. See
  /// [`prove_shared_range_check`].
  pub(crate) range_check: SharedRangeCheck,
  /// ONE combined opening discharging every evaluation claim made
  /// anywhere in the protocol, over all commitments in canonical order:
  /// the input `f`, the stacked layers `ab_1..ab_t`, the `1+2t` chunk
  /// commitments, and the multiplicity table.
  pub(crate) combined_open: CombinedBatchOpen,
}

/// ONE combined multi-point opening for ALL commitments of a Mod-PCS
/// open. Per commitment, its claims are RLC-combined (`Σ_i λ^i·y_i =
/// Σ_x f(x)·W(x)`); the per-commitment degree-2 sumchecks then run
/// INTERLEAVED with shared, tail-aligned challenges (every round all
/// active instances absorb their round polynomials before the one shared
/// challenge is squeezed — shorter instances join late). Each
/// commitment's final point is therefore the last `n_j` shared
/// challenges, so all final points share the same column coordinates and
/// the openings collapse into a single μ-RLC'd inner-product argument
/// ([`HyraxPCS::prove_same_column_batch`]). Sub-column-width commitments
/// (test sizes) fall back to individual opens.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct CombinedBatchOpen {
  /// Per-commitment compressed sumcheck round polynomials, tail-aligned
  /// to the shared challenge vector (entry `j` has `n_j` rounds).
  pub(crate) round_polys: Vec<Vec<crate::polys::univariate::CompressedUniPoly<t256::Scalar>>>,
  /// Per-commitment claimed final evaluation `f_j(r_j)`.
  pub(crate) final_evals: Vec<t256::Scalar>,
  /// The merged same-column opening (`f_y` is the μ-combined value
  /// `Σ μ^pos·f_j(r_j)` over the merged commitments). `None` iff no
  /// commitment reaches the column width.
  pub(crate) merged: Option<SmallPrimeOpening>,
  /// Individual fallback opens for sub-column-width commitments, in
  /// canonical order.
  pub(crate) small_opens: Vec<SmallPrimeOpening>,
}

/// Chunk width (bits) for the LogUp range checks: values are decomposed
/// into base-`2^16` chunks, each looked up against the `[0, 2^16)` table.
pub(crate) const CHUNK_BITS: usize = 16;

/// Per-batch data of the shared range check: the batch's chunk
/// commitment, the claimed evaluations its checks rest on (discharged by
/// the final batched opens), and the value-reconstruction sumcheck tying
/// chunks to the batch's value polynomials.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct RangeCheckBatchData {
  /// Stacked chunk-polynomial commitment (entries in `[0, 2^16)`),
  /// laid out `((p·n_values + within)·stride + c)`.
  pub(crate) chunk_comm: <Hyrax as PCSEngineTrait<T256HyraxEngine>>::Commitment,
  /// Claimed `V(r_v) = Σ_p eq(r_v_poly, p)·value_p(r_v_within)` — for an
  /// `a/b` batch this equals the stacked layer MLE at `(role, r_v)`, for
  /// the `f_limb` batch it is `f(r_v_within)`; discharged by the
  /// corresponding commitment's batched open.
  pub(crate) value_eval: t256::Scalar,
  /// Claimed `chunk(r_v ++ r_b)` — the reconstruction sumcheck's final
  /// chunk evaluation, discharged by the chunk commitment's batched open.
  pub(crate) reconstr_eval: t256::Scalar,
  /// Value-reconstruction sumcheck (`Σ_c 2^(16c)·chunk(r_v, c) =
  /// value(r_v)`), over the Hyrax base field.
  pub(crate) value_reconstr_sumcheck: crate::sumcheck::SumcheckProof<T256HyraxEngine>,
}

/// ONE shared LogUp-GKR range check covering all `(bound, size)` batch
/// groups of a Mod-PCS opening. Every batch's 16-bit chunks (and, for
/// non-16-aligned bounds, its shifted top chunks) are witness trees of a
/// single [`crate::logup_gkr::LogUpMultiRangeProof`] against one
/// `2^16`-entry multiplicity table — the table-side GKR and the
/// multiplicity commitment are paid once per opening instead of once per
/// batch. Witness-tree order: all batches' chunk trees in canonical
/// batch order, then the shifted-top trees of the non-aligned batches in
/// the same order.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct SharedRangeCheck {
  /// Commitment to the shared `2^16`-entry multiplicity table.
  pub(crate) mult_comm: <Hyrax as PCSEngineTrait<T256HyraxEngine>>::Commitment,
  /// The multi-witness LogUp-GKR membership argument.
  pub(crate) logup: crate::logup_gkr::LogUpMultiRangeProof<T256HyraxEngine>,
  /// Per-batch commitments, openings, and reconstruction sumchecks, in
  /// canonical batch order (`f_limb`, then `a_j`/`b_j` per iteration).
  pub(crate) batches: Vec<RangeCheckBatchData>,
}

/// `BigUint → t256::Scalar` via 64-byte wide reduction. Value-preserving
/// for inputs below the scalar field, otherwise reduces uniformly. Phase 3
/// step D will add the range check that turns this into a *sound*
/// commitment to a bounded integer.
fn biguint_to_scalar(v: &BigUint) -> t256::Scalar {
  let mut bytes = v.to_bytes_le();
  bytes.resize(64, 0);
  <t256::Scalar as PrimeFieldExt>::from_uniform(&bytes)
}

/// `t256::Scalar → BigUint` via the canonical (non-Montgomery) integer
/// representation. Inverse of `biguint_to_scalar` for inputs that fit
/// in the scalar field.
fn scalar_to_biguint(s: &t256::Scalar) -> BigUint {
  BigUint::from_bytes_le(s.to_repr().as_ref())
}

/// `t256::Scalar → BigInt` in *balanced* representation. The canonical
/// integer in `[0, q)` is reinterpreted as a signed integer in
/// `[-q/2, q/2)`: values `≥ ⌈q/2⌉` become `value - q`. Used by the
/// IntEval CRT check: when the integer evaluation is negative, the F
/// arithmetic produces a result near `q` (because `(1 - r_i)` wraps to
/// `q + 1 - r_i`), so the verifier must lift back to a signed value.
fn scalar_to_balanced_int(s: &t256::Scalar) -> BigInt {
  let v = scalar_to_biguint(s);
  let q = t256_q();
  let half = &q >> 1;
  if v > half {
    BigInt::from(v) - BigInt::from(q)
  } else {
    BigInt::from(v)
  }
}

/// The T256 scalar field's characteristic `q` as a `BigUint`. Computed
/// once via `(q - 1) + 1` from `-Scalar::ONE`'s representation; cheap
/// enough to recompute per call since it's just byte arithmetic.
fn t256_q() -> BigUint {
  // `q` is a compile-time constant of the curve; compute it once and
  // hand out clones (this is called per `shift_b`, i.e. per range check).
  static Q: once_cell::sync::Lazy<BigUint> = once_cell::sync::Lazy::new(|| {
    let q_minus_1 = (-<t256::Scalar as Field>::ONE).to_repr();
    let mut bytes = q_minus_1.as_ref().to_vec();
    let mut carry = 1u8;
    for b in bytes.iter_mut() {
      let (v, c) = b.overflowing_add(carry);
      *b = v;
      carry = u8::from(c);
    }
    debug_assert_eq!(carry, 0);
    BigUint::from_bytes_le(&bytes)
  });
  Q.clone()
}

/// Canonical integer in `[0, p)` from a `DynPrime<4>` value.
fn dyn_to_biguint(d: &crate::dyn_prime::DynPrime<4>) -> BigUint {
  BigUint::from_bytes_le(&d.to_le_bytes())
}

/// Extract `p` (the dynamic prime) from a non-empty point. Uses the
/// modulus carried by the first component's `FixedMontyParams<4>`.
fn extract_p(point: &[crate::dyn_prime::DynPrime<4>]) -> Result<BigUint, SpartanError> {
  let p0 = point.first().ok_or(SpartanError::InternalError {
    reason: "IntegerModPCS: point must have at least one component to extract p".to_string(),
  })?;
  let modulus = p0.params().modulus();
  // `modulus` is `&Odd<Uint<4>>`; `.as_ref()` gives the inner `Uint<4>`.
  let bytes = modulus.as_ref().to_le_bytes();
  Ok(BigUint::from_bytes_le(bytes.as_slice()))
}

/// Number of limbs needed to represent any value bounded by `T_f`
/// using limbs each bounded by `T`: `numlimb = ⌈log_T(T_f)⌉ = ⌈log_t_f
/// / log_t⌉`. Returns `1` for the no-limb-split degenerate case
/// (`log_t == log_t_f`).
pub fn numlimb(log_t_f: usize, log_t: usize) -> usize {
  assert!(log_t > 0, "log_t must be positive");
  log_t_f.div_ceil(log_t).max(1)
}

/// Bit-width of the limb index — `⌈log_2 numlimb⌉`. `0` if
/// `numlimb == 1` (no extra polynomial variables needed).
pub fn numlimb_var(numlimb: usize) -> usize {
  ceil_log2(numlimb.max(1))
}

/// Decompose a `BigUint` value `v ∈ [0, 2^log_bound)` into base-`2^16`
/// little-endian chunks: `v = sum_c 2^(16c) · chunks[c]` with
/// `chunks[c] < 2^16` and `⌈log_bound / 16⌉` entries. Asserts
/// `v < 2^log_bound`; values that exceed the bound are caller errors.
/// Used by the LogUp range-check arguments.
fn chunk_decompose_value(v: &BigUint, log_bound: usize) -> Vec<u64> {
  let numchunks = log_bound.div_ceil(CHUNK_BITS);
  let bytes = v.to_bytes_le();
  debug_assert!(
    bit_decompose_check_no_overflow(&bytes, log_bound),
    "value 0x{:x} exceeds bound 2^{}",
    v,
    log_bound
  );
  let byte_at = |i: usize| -> u64 { if i < bytes.len() { bytes[i] as u64 } else { 0 } };
  (0..numchunks)
    .map(|c| byte_at(2 * c) | (byte_at(2 * c + 1) << 8))
    .collect()
}

/// Helper for `chunk_decompose_value`'s debug_assert: checks that the
/// LE `bytes` representation has zero bits above `num_bits`.
fn bit_decompose_check_no_overflow(bytes: &[u8], num_bits: usize) -> bool {
  let cutoff_byte = num_bits / 8;
  let cutoff_bit = num_bits % 8;
  for (i, b) in bytes.iter().enumerate() {
    match i.cmp(&cutoff_byte) {
      std::cmp::Ordering::Less => {}
      std::cmp::Ordering::Equal => {
        if cutoff_bit < 8 && (*b >> cutoff_bit) != 0 {
          return false;
        }
      }
      std::cmp::Ordering::Greater => {
        if *b != 0 {
          return false;
        }
      }
    }
  }
  true
}

/// Big-endian boolean MLE point for the index `idx` over `num_bits`
/// variables: `point[0]` is the most significant bit. Binding an MLE's
/// trailing variables to this point selects the slot `idx` of the
/// bottom axis.
fn bool_point_of_index(idx: usize, num_bits: usize) -> Vec<t256::Scalar> {
  (0..num_bits)
    .rev()
    .map(|b| {
      if (idx >> b) & 1 == 1 {
        t256::Scalar::ONE
      } else {
        t256::Scalar::ZERO
      }
    })
    .collect()
}

/// Split a single `BigUint` value `v ∈ [0, 2^log_t_f)` into `numlimb`
/// limbs each in `[0, 2^log_t)`, base-`T` little-endian: `v = sum_i
/// T^i · limbs[i]`. Asserts `v < 2^(numlimb · log_t)`; values that
/// exceed the declared bound `T_f` are caller errors (Phase 3 step D5
/// adds the soundness-grade range check).
fn split_value_into_limbs(v: &BigUint, log_t: usize, numlimb: usize) -> Vec<BigUint> {
  let t = BigUint::one() << log_t;
  let mut out = Vec::with_capacity(numlimb);
  let mut rem = v.clone();
  for _ in 0..numlimb {
    let (q, r) = (&rem / &t, &rem % &t);
    out.push(r);
    rem = q;
  }
  debug_assert!(
    rem.is_zero(),
    "value 0x{:x} exceeds bound 2^{}",
    v,
    numlimb * log_t
  );
  out
}

/// Build the public limb-weight polynomial `limb` as a `DynPrime<4>`
/// MLE of size `2^numlimb_var`: `limb[k] = T^k` for `k < numlimb`, else
/// `0` (padding when `numlimb` isn't a power of two). Used by the
/// Phase-3 step D3 reduction sumcheck integrand
/// `sum_k limb(k) · f_limb(int_r, k)`.
fn build_limb_weight_dynprime(
  params: &IntEvalParams,
  monty: &crypto_bigint::modular::FixedMontyParams<4>,
) -> Vec<crate::dyn_prime::DynPrime<4>> {
  let stride = 1usize << params.numlimb_var;
  let t = BigUint::one() << params.log_t;
  let mut out = Vec::with_capacity(stride);
  let mut pow = BigUint::one();
  for k in 0..stride {
    if k < params.numlimb {
      out.push(
        <crate::dyn_prime::DynPrime<4> as SumcheckField>::from_bytes_reduce(
          monty,
          &pow.to_bytes_le(),
        ),
      );
      pow = &pow * &t;
    } else {
      out.push(<crate::dyn_prime::DynPrime<4> as SumcheckField>::zero(
        monty,
      ));
    }
  }
  out
}

/// Limb-split a multilinear polynomial. Input `poly` has length `2^n`;
/// output has length `2^n · 2^numlimb_var` where `numlimb_var =
/// ⌈log_2 numlimb⌉`. Layout: `f_limb[x · 2^numlimb_var + k]` is the
/// `k`-th limb of `f[x]` for `k < numlimb`, else `0`. The original
/// `n` variables occupy the top bits of the combined index, limb
/// variables the bottom bits — matches `EqPolynomial::evals_from_points`'s
/// convention so the limb-reduction sumcheck (step D3) treats the
/// limb dimension as the *last* variables.
fn limb_split_polynomial(poly: &[BigUint], log_t: usize, log_t_f: usize) -> Vec<BigUint> {
  let numlimb = numlimb(log_t_f, log_t);
  let numlimb_var = numlimb_var(numlimb);
  let stride = 1usize << numlimb_var;
  // Each coefficient expands to `stride` contiguous slots: its `numlimb`
  // limbs followed by zero padding. Order is preserved across the
  // parallel map, so the `x · stride + k` layout is identical to the
  // sequential version.
  poly
    .par_iter()
    .flat_map_iter(|v| {
      let mut limbs = split_value_into_limbs(v, log_t, numlimb);
      limbs.resize(stride, BigUint::zero());
      limbs.into_iter()
    })
    .collect()
}

/// Truncated (toward-zero) divmod. Returns `(q, r)` with `q · d + r = g`
/// and `sign(r) = sign(g)` (or `r = 0`); `|r| < d`. Used by IntEval's
/// partial-evaluation decomposition for *symmetric* remainder/quotient,
/// matching the user-preferred convention (deviates from the paper's
/// floor-division `⌊·/p_i⌋`). The integer identity `a + p · b = g`
/// holds the same for both, so soundness is unchanged.
fn truncated_divmod(g: &BigInt, d: &BigUint) -> (BigInt, BigInt) {
  let d_big = BigInt::from(d.clone());
  let q = g / &d_big;
  let r = g - &q * &d_big;
  (q, r)
}

/// Public shift bound for an `a_j` polynomial: under truncated divmod
/// with divisor `p_i`, `a_j(x) ∈ (-p_i, p_i)`. Using the universal
/// upper bound `P = 2^log_p` for all primes in the sample range gives
/// a constant shift per-`params`, independent of the specific `p_i`.
fn shift_a(params: &IntEvalParams) -> BigUint {
  BigUint::one() << params.log_p
}

/// Public shift bound for a `b_j` polynomial: per the paper's bound
/// `||g_j|| < (q-P)/2` and `|b_j| ≤ ||g_j||/p_i`, we have
/// `|b_j| < (q-P)/(2 p_i) < q/(2·P/2) = q/P` (using `p_i ≥ P/2`).
/// So shifting by `⌊q/P⌋` is sound. Like `shift_a`, this is a public
/// per-`params` constant.
fn shift_b(params: &IntEvalParams) -> BigUint {
  &t256_q() / (BigUint::one() << params.log_p)
}

/// Integer partial-evaluation at the *last* `k` variables. Given a
/// multilinear polynomial `poly` of `2^n_cur` evaluations and a binding
/// vector `r_lower` of length `k`, returns the `2^(n_cur - k)`
/// evaluations of `g(X) = poly(X, r_lower)`. Computed over Z (no
/// reduction); intermediate magnitudes can grow large.
fn integer_partial_evaluate_top_k(poly: &[BigInt], r_lower: &[BigUint]) -> Vec<BigInt> {
  let k = r_lower.len();
  let two_k = 1usize << k;
  assert!(poly.len().is_multiple_of(two_k));
  let new_size = poly.len() / two_k;

  let r_int: Vec<BigInt> = r_lower.iter().map(|x| BigInt::from(x.clone())).collect();
  let one = BigInt::one();

  // Precompute integer chi(r_lower, y) for y ∈ [0, 2^k). Bit-order
  // matches `EqPolynomial::evals_from_points`: variable i corresponds
  // to bit (k-1-i) of y.
  let chi_table: Vec<BigInt> = (0..two_k)
    .map(|y| {
      let mut chi = one.clone();
      for (i, ri) in r_int.iter().enumerate().take(k) {
        let bit = (y >> (k - 1 - i)) & 1;
        let factor = if bit == 1 { ri.clone() } else { &one - ri };
        chi *= factor;
      }
      chi
    })
    .collect();

  (0..new_size)
    .into_par_iter()
    .map(|x| {
      let mut slot = BigInt::zero();
      for (y, chi_y) in chi_table.iter().enumerate().take(two_k) {
        slot += &poly[x * two_k + y] * chi_y;
      }
      slot
    })
    .collect()
}

/// Compute the signed integer MLE evaluation `sum_k chi_int(k, point) ·
/// poly[k]`, where `chi_int(k, point) = prod_i (k_i · point_i + (1-k_i) ·
/// (1-point_i))` over Z (no reduction). Returns the full integer.
///
/// Used by the IntEval prover to compute `int_v' = f(int_r)`. The result
/// can be huge — bounded by `2^n · p^n · max(|poly|)` in magnitude — and
/// can be negative when `(1 - point_i)` flips signs.
fn integer_mle_evaluate(poly: &[BigUint], point: &[BigUint]) -> BigInt {
  let n = poly.len();
  let num_vars = n.trailing_zeros() as usize;
  debug_assert_eq!(1 << num_vars, n);
  debug_assert_eq!(point.len(), num_vars);

  // Bind variables one at a time (top variable first), folding the table
  // in half per round over ℤ: out[i] = lo[i] + r·(hi[i] − lo[i]). The
  // bit-order matches `EqPolynomial::evals_from_points` / `bind_poly_var_top`:
  // variable `i` corresponds to bit `num_vars - 1 - i` of the index, so
  // `point[0]` splits the table into its two halves. Entry widths grow by
  // ~|r| bits per round while the count halves, so the total bigint work
  // is dominated by the early (narrow) rounds — far cheaper than
  // materializing the 2^n chi table at full width, and each round is
  // embarrassingly parallel.
  let mut cur: Vec<BigInt> = poly.par_iter().map(|x| BigInt::from(x.clone())).collect();
  for r in point {
    let r_int = BigInt::from(r.clone());
    let h = cur.len() / 2;
    let (lo, hi) = cur.split_at(h);
    cur = if h >= 1 << 10 {
      lo.par_iter()
        .zip(hi.par_iter())
        .map(|(a, b)| a + &r_int * (b - a))
        .collect()
    } else {
      lo.iter()
        .zip(hi.iter())
        .map(|(a, b)| a + &r_int * (b - a))
        .collect()
    };
  }
  cur.pop().expect("non-empty table")
}

/// Rejection-sample a small prime in `[2^{log_p - 1}, 2^{log_p})` from
/// the transcript via Miller-Rabin / Lucas BPSW. Squeezes 64 bytes at a
/// time, builds a `log_p`-bit candidate with the MSB and LSB forced,
/// runs `crypto_primes::is_prime`, and retries on composite. The two
/// sides (prover & verifier) drive the transcript identically, so they
/// arrive at the same prime.
fn sample_small_prime<T: ByteTranscript>(
  transcript: &mut T,
  log_p: usize,
) -> Result<BigUint, SpartanError> {
  use crypto_primes::{Flavor, is_prime};
  // `crypto_primes::is_prime` works over `Uint<L>`; we use `U256` here
  // since `log_p` is bounded by `LOG_Q = 256`.
  use crypto_bigint::U256;
  assert!(log_p > 1 && log_p <= LOG_Q);
  let bytes_needed = log_p.div_ceil(8);
  loop {
    let bytes = transcript.squeeze_bytes(b"sample_small_p")?;
    let mut buf = [0u8; 32];
    buf[..bytes_needed].copy_from_slice(&bytes[..bytes_needed]);
    // Force MSB of bit (log_p - 1) so candidate has exactly log_p bits;
    // force LSB so it's odd. Clear bits above log_p - 1 so width is exact.
    let top_byte = (log_p - 1) / 8;
    let top_bit_in_byte = (log_p - 1) % 8;
    // Clear bits above log_p - 1.
    if top_byte < 32 {
      let mask_top: u8 = (1u16 << (top_bit_in_byte + 1)).wrapping_sub(1) as u8;
      buf[top_byte] &= mask_top;
      for b in &mut buf[(top_byte + 1)..] {
        *b = 0;
      }
    }
    // Force MSB and LSB.
    buf[top_byte] |= 1u8 << top_bit_in_byte;
    buf[0] |= 0x01;
    let candidate = U256::from_le_slice(&buf);
    if is_prime(Flavor::Any, &candidate) {
      return Ok(BigUint::from_bytes_le(&buf));
    }
  }
}

/// Sound Mod-PCS for `T256DynPrimeEngine`. See module docs.
#[derive(Clone)]
pub struct IntegerModPCS {
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

impl IntegerModPCS {
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
    // Hyrax CK must be sized for the *limb-split* polynomial: the
    // input poly has `n` coefficients, but after limb-splitting each
    // coefficient becomes `2^numlimb_var` slots. For numlimb_var=0
    // (no-limb-split) this is `n` unchanged.
    let inflated_n =
      n.checked_shl(params.numlimb_var as u32)
        .ok_or(SpartanError::InvalidInputLength {
          reason: format!(
            "n={n} * 2^numlimb_var={} overflows usize",
            params.numlimb_var
          ),
        })?;
    let (inner_ck, inner_vk) = Hyrax::setup(label, inflated_n, width);
    // Size-1 eval key for the internal `G^{f_y}` eval commitments (kept
    // inside the Mod-PCS key, off the PCS-agnostic trait surface).
    let (eval_ck, _) = Hyrax::setup(b"imod_modpcs_eval", 1, 1);
    // Precompute once at setup so neither prove nor verify rebuilds the
    // eval key's generator table per `G^{f_y}` (re)construction. Cloning
    // after precompute propagates it into both the commitment and verifier
    // keys (mirrors the pre-refactor `precompute_ck(&ck_s)`).
    Hyrax::precompute_ck(&eval_ck);
    Ok((
      IntegerModCommitmentKey {
        inner: inner_ck,
        eval: eval_ck.clone(),
        params: params.clone(),
      },
      IntegerModVerifierKey {
        inner: inner_vk,
        eval: eval_ck,
        params,
      },
    ))
  }

  /// Like the trait `setup`, but instead of the fixed application
  /// defaults `(DEFAULT_LOG_T_F, DEFAULT_K)` it derives params
  /// *optimized for this input length* via
  /// [`IntEvalParams::derive_optimized`]: `(k, log T, log P, s)` are
  /// chosen to minimize the estimated prover cost for an `n`-coefficient
  /// polynomial with coefficients bounded by `2^log_t_f`. The chosen
  /// params still pass the full §4.4 `validate`, so this never trades
  /// soundness for speed.
  pub fn setup_optimized(
    label: &'static [u8],
    n: usize,
    width: usize,
    log_t_f: usize,
  ) -> Result<
    (
      <Self as ModPCSEngineTrait<T256DynPrimeEngine>>::CommitmentKey,
      <Self as ModPCSEngineTrait<T256DynPrimeEngine>>::VerifierKey,
    ),
    SpartanError,
  > {
    let num_vars = ceil_log2(n.max(1));
    let params = IntEvalParams::derive_optimized(log_t_f, num_vars)?;
    Self::setup_with_params(label, n, width, params)
  }
}

impl ModPCSEngineTrait<T256DynPrimeEngine> for IntegerModPCS {
  type CommitmentKey = IntegerModCommitmentKey;
  type VerifierKey = IntegerModVerifierKey;
  type Commitment = IntegerModCommitment;
  type Blind = IntegerModBlind;
  type EvaluationArgument = IntEvalArgument;

  /// Trait-driven setup: derive `IntEvalParams` optimized for this
  /// polynomial size via [`IntEvalParams::derive_optimized`], with the
  /// application-default norm bound `DEFAULT_LOG_T_F`. Panics if the
  /// derivation fails (which only happens for pathologically small `n`);
  /// callers that need control over the security or norm-bound
  /// parameters should use `setup_with_params`.
  fn setup(
    label: &'static [u8],
    n: usize,
    width: usize,
  ) -> (Self::CommitmentKey, Self::VerifierKey) {
    let num_vars = ceil_log2(n.max(1));
    let params = IntEvalParams::derive_optimized(DEFAULT_LOG_T_F, num_vars).expect(
      "default IntEvalParams derivation must satisfy the paper's bounds; \
         override with `setup_with_params` to use tighter parameters",
    );
    let inflated_n = n
      .checked_shl(params.numlimb_var as u32)
      .expect("n * 2^numlimb_var overflows usize");
    let (inner_ck, inner_vk) = Hyrax::setup(label, inflated_n, width);
    let (eval_ck, _) = Hyrax::setup(b"imod_modpcs_eval", 1, 1);
    // Precompute the eval key once at setup (see `setup_with_params`).
    Hyrax::precompute_ck(&eval_ck);
    (
      IntegerModCommitmentKey {
        inner: inner_ck,
        eval: eval_ck.clone(),
        params: params.clone(),
      },
      IntegerModVerifierKey {
        inner: inner_vk,
        eval: eval_ck,
        params,
      },
    )
  }

  fn precompute_ck(ck: &Self::CommitmentKey) {
    Hyrax::precompute_ck(&ck.inner);
    // Also precompute the size-1 eval key (used for the internal `G^{f_y}`
    // eval commitments) so a deserialized key doesn't rebuild it lazily.
    Hyrax::precompute_ck(&ck.eval);
  }

  fn blind(ck: &Self::CommitmentKey, n: usize) -> Self::Blind {
    // `commit` limb-splits an `n`-coefficient polynomial to
    // `2^numlimb_var · n` coefficients before reaching the inner Hyrax
    // PCS, so the blind must cover that inflated length. For
    // `numlimb_var = 0` (no-limb-split) this is `n` unchanged. (Size-1
    // eval commits use a `ck_eval` with `numlimb_var = 0`, and `commit`
    // skips splitting for `v.len() == 1`, so they are unaffected.)
    let inflated = n << ck.params.numlimb_var;
    IntegerModBlind {
      inner: Hyrax::blind(&ck.inner, inflated),
    }
  }

  fn commit(
    ck: &Self::CommitmentKey,
    v: &[BigUint],
    r: &Self::Blind,
  ) -> Result<Self::Commitment, SpartanError> {
    // Limb-split the integer-valued polynomial: each coefficient
    // becomes `numlimb` limbs each in `[0, T)`. For `numlimb = 1`
    // (no-limb-split mode) this returns `v` unchanged. The underlying
    // F PCS commits the limb polynomial, which has `numlimb * v.len()`
    // (or `2^numlimb_var * v.len()` after padding) coefficients.
    //
    // Stopgap: size-1 commits are single-value commits used internally
    // (e.g. the eval-value commitment); the value may be any F element,
    // not bounded by `T_f`, so skip limb-splitting in that case.
    //
    // The small-scalar MSM fast path (`is_small`) requires every committed
    // value to be `< 2^64` — Hyrax takes the low 8 bytes with no check, so a
    // wider value would be silently truncated to a wrong commitment. Limbs
    // are `< 2^log_t`, so claim `is_small` only when `log_t <= 64`; wider
    // limbs (e.g. a `log_t = 128` choice from `derive_optimized`) take the
    // checked full-MSM path. The size-1 stopgap value may be a full-width
    // `Z_p` element, so `is_small = false` there too.
    //
    // TODO Phase 3 step D5: per-limb range check `|limb| < T`.
    let params = &ck.params;
    let (v_limbs, is_small) = if v.len() == 1 {
      (v.to_vec(), false)
    } else {
      (
        limb_split_polynomial(v, params.log_t, params.log_t_f),
        params.log_t <= 64,
      )
    };
    let v_fq: Vec<t256::Scalar> = v_limbs.iter().map(biguint_to_scalar).collect();
    let inner = Hyrax::commit(&ck.inner, &v_fq, &r.inner, is_small)?;
    Ok(IntegerModCommitment { inner })
  }

  fn check_commitment(comm: &Self::Commitment, n: usize, width: usize) -> Result<(), SpartanError> {
    Hyrax::check_commitment(&comm.inner, n, width)
  }

  fn prove(
    ck: &Self::CommitmentKey,
    transcript: &mut <T256DynPrimeEngine as SumcheckEngine>::TE,
    comm: &Self::Commitment,
    poly: &[BigUint],
    blind: &Self::Blind,
    point: &[<T256DynPrimeEngine as SumcheckEngine>::Scalar],
    eval: &BigUint,
  ) -> Result<Self::EvaluationArgument, SpartanError> {
    let (_prove_span, prove_t) = start_span!("integer_modpcs_prove");
    let params = &ck.params;
    let monty = point
      .first()
      .map(|p| *p.params())
      .ok_or(SpartanError::InternalError {
        reason: "IntegerModPCS::prove: empty point".to_string(),
      })?;

    let (_red_span, red_t) = start_span!("imod_pcs_reduction_sc");
    // 0. Limb-split f → f_limb. For numlimb=1 this is a literal pass-
    //    through; for numlimb>1 f_limb has 2^numlimb_var times as many
    //    coefficients (and 2^numlimb_var slots per original coefficient,
    //    padded with zero if numlimb isn't a power of two).
    let (_ls_span, ls_t) = start_span!("imod_pcs_red_limb_split");
    let f_limb = limb_split_polynomial(poly, params.log_t, params.log_t_f);
    info!(elapsed_ms = %ls_t.elapsed().as_millis(), "imod_pcs_red_limb_split");

    // 1. Reduction sumcheck (Phase-3 step D3): reduce the eval claim
    //    `f(int_r) ≡_p eval` to a claim about `f_limb` at a combined
    //    point `(int_r, r_k)` where `r_k` are the sumcheck challenges.
    //    Integrand: `limb(k) · f_limb(int_r, k)`, summed over k ∈
    //    {0,1}^numlimb_var. For numlimb_var = 0 the sumcheck has zero
    //    rounds, returns `r_k = []`, and the recovered eval equals
    //    the input `eval` directly (limb(empty) = T^0 = 1).
    let (_dc_span, dc_t) = start_span!("imod_pcs_red_to_dynprime");
    let f_limb_p: Vec<crate::dyn_prime::DynPrime<4>> = f_limb
      .iter()
      .map(|b| {
        <crate::dyn_prime::DynPrime<4> as SumcheckField>::from_bytes_reduce(
          &monty,
          &b.to_bytes_le(),
        )
      })
      .collect();
    info!(elapsed_ms = %dc_t.elapsed().as_millis(), "imod_pcs_red_to_dynprime");
    // Partial-eval f_limb at the original `point` in Z_p, leaving the
    // last numlimb_var variables free. Bind the top variables (= the
    // original n_vars) one at a time.
    let (_pe_span, pe_t) = start_span!("imod_pcs_red_dynprime_bind");
    let mut mle = crate::polys_modp::multilinear::MultilinearPolynomial::new(f_limb_p, monty);
    for r_i in point {
      mle.bind_poly_var_top(r_i);
    }
    let f_limb_at_int_r: Vec<crate::dyn_prime::DynPrime<4>> = mle.into_vec();
    info!(elapsed_ms = %pe_t.elapsed().as_millis(), "imod_pcs_red_dynprime_bind");
    debug_assert_eq!(f_limb_at_int_r.len(), 1 << params.numlimb_var);

    let limb_p = build_limb_weight_dynprime(params, &monty);
    let eval_p = <crate::dyn_prime::DynPrime<4> as SumcheckField>::from_bytes_reduce(
      &monty,
      &eval.to_bytes_le(),
    );

    let mut poly_lhs = crate::polys_modp::multilinear::MultilinearPolynomial::new(limb_p, monty);
    let mut poly_rhs =
      crate::polys_modp::multilinear::MultilinearPolynomial::new(f_limb_at_int_r, monty);
    let (red_sc, r_k, final_claims) =
      crate::sumcheck_modp::SumcheckProof::<T256DynPrimeEngine>::prove_quad(
        &eval_p,
        params.numlimb_var,
        &mut poly_lhs,
        &mut poly_rhs,
        transcript,
      )?;
    // `final_claims = [limb(r_k), f_limb(int_r, r_k)]`. The integer-side
    // IntEval will prove `int_v' ≡ f_limb(int_r, r_k) (mod p)`, so the
    // "eval claim" handed to the IntEval body is the second component.
    let f_eval_p = final_claims[1];

    // 2. Extend the integer point with r_k (canonical < p integers).
    let int_point: Vec<BigUint> = point.iter().map(dyn_to_biguint).collect();
    let r_k_int: Vec<BigUint> = r_k.iter().map(dyn_to_biguint).collect();
    let int_point_ext: Vec<BigUint> = int_point.iter().chain(r_k_int.iter()).cloned().collect();

    // 3. int_v' = f_limb at the extended point, over Z.
    let (_ie_span, ie_t) = start_span!("imod_pcs_red_integer_eval");
    let int_v_prime = integer_mle_evaluate(&f_limb, &int_point_ext);
    info!(elapsed_ms = %ie_t.elapsed().as_millis(), "imod_pcs_red_integer_eval");

    // 4. Sanity: f_eval_p ≡ int_v' (mod p). For numlimb_var=0, f_eval_p
    //    == eval and this matches the pre-D3 check.
    let p = extract_p(point)?;
    let int_v_mod_p_u = int_v_prime
      .mod_floor(&BigInt::from(p.clone()))
      .to_biguint()
      .expect("mod_floor by a positive divisor is non-negative");
    let f_eval_bu = BigUint::from_bytes_le(&f_eval_p.to_le_bytes());
    if int_v_mod_p_u != f_eval_bu {
      return Err(SpartanError::InternalError {
        reason: "IntegerModPCS::prove: f_limb(ext_point) ≠ int_v' mod p (prover bug)".to_string(),
      });
    }

    // 5. Bind int_v' into the transcript.
    absorb_bigint(transcript, &int_v_prime);

    // Extract round polys from the reduction sumcheck as a Serde-
    // friendly BigUint payload for the verifier.
    let reduction_round_polys: Vec<Vec<BigUint>> = red_sc
      .compressed_polys
      .iter()
      .map(|cp| {
        cp.coeffs_except_linear_term
          .iter()
          .map(dyn_to_biguint)
          .collect()
      })
      .collect();

    // From here on the chain prover operates on `f_limb` over the
    // extended point. For numlimb_var=0 these match the pre-D3 `poly`
    // / `point` exactly.
    let int_point = int_point_ext;
    let num_vars = point.len() + params.numlimb_var;
    let with_iter = num_vars > params.k;
    let poly = f_limb.as_slice();
    let (_fq_span, fq_t) = start_span!("imod_pcs_red_to_fq");
    let poly_fq: Vec<t256::Scalar> = poly.iter().map(biguint_to_scalar).collect();
    info!(elapsed_ms = %fq_t.elapsed().as_millis(), "imod_pcs_red_to_fq");
    info!(elapsed_ms = %red_t.elapsed().as_millis(), "imod_pcs_reduction");

    // 4. Phase 1: per prime, sample p_i, run all t iterations (if any).
    //    No per-chain commitments: all chains' shifted (a_j, b_j) are
    //    stacked into ONE commitment per layer below.
    let (_p1_span, p1_t) = start_span!("imod_pcs_chain_phase1");
    // Lift `poly` to BigInt once; each iterating chain clones it as its
    // `a_0`. Only the `with_iter` path consumes it.
    let poly_bigint: Vec<BigInt> = if with_iter {
      poly.par_iter().map(|x| BigInt::from(x.clone())).collect()
    } else {
      Vec::new()
    };

    // Sample all `s` small primes upfront (paper §4: the verifier sends
    // `[p_i]_{i=1}^s` in one round, then the prover responds with the chain
    // oracles). The chains are independent and built in parallel.
    let primes: Vec<BigUint> = (0..params.s)
      .map(|_| sample_small_prime(transcript, params.log_p))
      .collect::<Result<Vec<_>, SpartanError>>()?;

    let chain_states: Vec<ChainProverState> = primes
      .par_iter()
      .map(|p_i| -> Result<ChainProverState, SpartanError> {
        let r_i_int: Vec<BigUint> = int_point.iter().map(|x| x % p_i).collect();
        let mut iters = Vec::new();

        if with_iter {
          let t = num_vars.saturating_sub(params.k).div_ceil(params.k);
          let n = num_vars;
          let k = params.k;
          let d_big = BigInt::from(p_i.clone());
          let s_a = BigInt::from(shift_a(params));
          let s_b = BigInt::from(shift_b(params));

          // a_prev starts as the input polynomial (lifted once above).
          let mut a_prev_int: Vec<BigInt> = poly_bigint.clone();

          for j in 1..=t {
            let lo = n - j * k;
            let hi = n - (j - 1) * k;
            let r_lower = &r_i_int[lo..hi];

            let g_j_int = integer_partial_evaluate_top_k(&a_prev_int, r_lower);
            // Toward-zero divmod by p_i: `q·p_i + r = g`, `(b, a) = (q, r)`.
            // Inner loop kept serial — the per-chain `par_iter` already
            // saturates the cores.
            let (b_j_int, a_j_int): (Vec<BigInt>, Vec<BigInt>) = g_j_int
              .iter()
              .map(|g| {
                let q = g / &d_big;
                let r = g - &q * &d_big;
                (q, r)
              })
              .unzip();

            let a_j_shifted: Vec<BigUint> = a_j_int
              .iter()
              .map(|x| (x + &s_a).to_biguint().expect("shift makes non-negative"))
              .collect();
            let b_j_shifted: Vec<BigUint> = b_j_int
              .iter()
              .map(|x| (x + &s_b).to_biguint().expect("shift makes non-negative"))
              .collect();

            let a_j_shifted_fq: Vec<t256::Scalar> =
              a_j_shifted.iter().map(biguint_to_scalar).collect();
            let b_j_shifted_fq: Vec<t256::Scalar> =
              b_j_shifted.iter().map(biguint_to_scalar).collect();

            iters.push(IterationProverState {
              a_shifted: a_j_shifted,
              a_shifted_fq: a_j_shifted_fq,
              b_shifted: b_j_shifted,
              b_shifted_fq: b_j_shifted_fq,
            });

            a_prev_int = a_j_int;
          }
        }

        Ok(ChainProverState {
          p_i: p_i.clone(),
          r_i_int,
          iters,
        })
      })
      .collect::<Result<Vec<_>, SpartanError>>()?;

    // Stack all chains' shifted (a_j, b_j) per layer into ONE commitment:
    // index = (role, chain, x), role 0 = a / 1 = b, chains padded to
    // s_pad = next_pow2(s). Absorbed in layer order, before γ.
    let t_layers = if with_iter {
      num_vars.saturating_sub(params.k).div_ceil(params.k)
    } else {
      0
    };
    let s_pad = params.s.next_power_of_two();
    let log_spad = s_pad.trailing_zeros() as usize;
    type HC = <Hyrax as PCSEngineTrait<T256HyraxEngine>>::Commitment;
    type HB = <Hyrax as PCSEngineTrait<T256HyraxEngine>>::Blind;
    let mut ab_polys: Vec<Vec<t256::Scalar>> = Vec::with_capacity(t_layers);
    let mut ab_blinds: Vec<HB> = Vec::with_capacity(t_layers);
    let mut ab_comms: Vec<HC> = Vec::with_capacity(t_layers);
    for jm1 in 0..t_layers {
      let m = 1usize << (num_vars - (jm1 + 1) * params.k);
      let mut stacked = vec![t256::Scalar::ZERO; 2 * s_pad * m];
      for (c, st) in chain_states.iter().enumerate() {
        let it = &st.iters[jm1];
        stacked[c * m..(c + 1) * m].copy_from_slice(&it.a_shifted_fq);
        stacked[(s_pad + c) * m..(s_pad + c + 1) * m].copy_from_slice(&it.b_shifted_fq);
      }
      let ab_blind = Hyrax::blind(&ck.inner, stacked.len());
      let ab_comm = Hyrax::commit(&ck.inner, &stacked, &ab_blind, false)?;
      transcript.absorb(b"ab_stacked", &ab_comm);
      ab_polys.push(stacked);
      ab_blinds.push(ab_blind);
      ab_comms.push(ab_comm);
    }
    info!(elapsed_ms = %p1_t.elapsed().as_millis(), "imod_pcs_chain_phase1");

    // 5. Sample γ ∈ F^{n-k} after all phase-1 commits are absorbed.
    let gamma_fq: Vec<t256::Scalar> = if with_iter {
      (0..(num_vars - params.k))
        .map(|i| {
          let bytes = transcript.squeeze_bytes(b"gamma")?;
          let label = (i as u64).to_le_bytes();
          transcript.absorb_bytes(b"gamma_idx", &label);
          Ok(<t256::Scalar as PrimeFieldExt>::from_uniform(&bytes))
        })
        .collect::<Result<Vec<_>, SpartanError>>()?
    } else {
      Vec::new()
    };

    // 6. Identity-check and final-remainder evaluations become multi-point
    //    CLAIMS on `f` / the stacked layer commitments; each claimed eval
    //    is absorbed (binding it before the batch challenges), and all
    //    claims are discharged by one batched open per commitment below.
    let (_open_span, open_t) = start_span!("imod_pcs_chain_claims");
    let mut f_claims = OpenClaims::default();
    let mut ab_claims: Vec<OpenClaims> = (0..t_layers).map(|_| OpenClaims::default()).collect();

    // Fast path for the j=1 a_prev evals: all chains share the γ-prefix,
    // so bind `f` once at γ[..n-k]; per chain only a 2^k-size dot with
    // eq(r_lower) remains.
    let poly_at_gamma: Vec<t256::Scalar> = if with_iter {
      let mut m = crate::polys::multilinear::MultilinearPolynomial::new(poly_fq.clone());
      for r in &gamma_fq[..(num_vars - params.k)] {
        m.bind_poly_var_top(r);
      }
      m.into_vec()
    } else {
      Vec::new()
    };

    let mut chains: Vec<ChainData> = Vec::with_capacity(params.s);
    for (ci, state) in chain_states.iter().enumerate() {
      let r_i_int = &state.r_i_int;
      let iters = &state.iters;
      let n = num_vars;
      let k = params.k;

      let mut iter_oracles = Vec::with_capacity(iters.len());
      for (jm1, iter_state) in iters.iter().enumerate() {
        let j = jm1 + 1;
        let prefix_len = n - j * k;
        let lo = n - j * k;
        let hi = n - (j - 1) * k;
        let r_lower_fq: Vec<t256::Scalar> = r_i_int[lo..hi].iter().map(biguint_to_scalar).collect();
        let gamma_prefix: Vec<t256::Scalar> = gamma_fq[..prefix_len].to_vec();
        let gamma_extended: Vec<t256::Scalar> = gamma_prefix
          .iter()
          .chain(r_lower_fq.iter())
          .copied()
          .collect();

        let a_prev_eval = if j == 1 {
          let v = mle_evaluate_fq(&poly_at_gamma, &r_lower_fq);
          f_claims.push(gamma_extended.clone(), v);
          v
        } else {
          let v = mle_evaluate_fq(&iters[jm1 - 1].a_shifted_fq, &gamma_extended);
          ab_claims[jm1 - 1].push(ab_stack_point(0, ci, log_spad, &gamma_extended), v);
          v
        };
        let a_curr_eval = mle_evaluate_fq(&iter_state.a_shifted_fq, &gamma_prefix);
        let b_curr_eval = mle_evaluate_fq(&iter_state.b_shifted_fq, &gamma_prefix);
        ab_claims[jm1].push(ab_stack_point(0, ci, log_spad, &gamma_prefix), a_curr_eval);
        ab_claims[jm1].push(ab_stack_point(1, ci, log_spad, &gamma_prefix), b_curr_eval);
        for ev in [&a_prev_eval, &a_curr_eval, &b_curr_eval] {
          transcript.absorb_bytes(b"claim_ev", ev.to_repr().as_ref());
        }
        iter_oracles.push(IterationOracles {
          a_prev_eval,
          a_curr_eval,
          b_curr_eval,
        });
      }

      // Final-remainder claim: a_t at r_i[0..n-tk]. For t=0 this is a
      // claim on the *input* polynomial at the full r_i (step B path).
      let t = iters.len();
      let final_point_fq: Vec<t256::Scalar> = r_i_int[..(num_vars - t * params.k)]
        .iter()
        .map(biguint_to_scalar)
        .collect();
      let final_eval = if t == 0 {
        let v = mle_evaluate_fq(&poly_fq, &final_point_fq);
        f_claims.push(final_point_fq, v);
        v
      } else {
        let last = &iters[t - 1];
        let v = mle_evaluate_fq(&last.a_shifted_fq, &final_point_fq);
        ab_claims[t - 1].push(ab_stack_point(0, ci, log_spad, &final_point_fq), v);
        v
      };
      transcript.absorb_bytes(b"claim_ev", final_eval.to_repr().as_ref());

      chains.push(ChainData {
        iterations: iter_oracles,
        final_eval,
      });
    }
    info!(elapsed_ms = %open_t.elapsed().as_millis(), "imod_pcs_chain_claims");

    // 7. ONE shared LogUp-GKR range check across all `(bound, size)`
    // groups, in canonical order `f_limb`, then for each iteration `j`
    // the `a_j` batch (all `s` chains) and the `b_j` batch. All batches
    // share one multiplicity table and one table-side GKR; its value and
    // chunk evaluations come back as claims.
    let t = t_layers;
    let log_bound_a = params.log_p + 1;
    let log_bound_b = LOG_Q - params.log_p + 1;
    let mut rc_batches: Vec<RangeBatchInputs<'_>> = Vec::with_capacity(1 + 2 * t);

    rc_batches.push(RangeBatchInputs {
      target: RcTarget::F,
      value_polys_fq: vec![poly_fq.as_slice()],
      values: vec![poly],
      n_values: poly.len(),
      log_bound: params.log_t,
    });

    for j in 0..t {
      for (role, log_bound) in [(0u8, log_bound_a), (1u8, log_bound_b)] {
        let value_polys_fq = chain_states
          .iter()
          .map(|st| {
            if role == 0 {
              st.iters[j].a_shifted_fq.as_slice()
            } else {
              st.iters[j].b_shifted_fq.as_slice()
            }
          })
          .collect::<Vec<_>>();
        let values = chain_states
          .iter()
          .map(|st| {
            if role == 0 {
              st.iters[j].a_shifted.as_slice()
            } else {
              st.iters[j].b_shifted.as_slice()
            }
          })
          .collect::<Vec<_>>();
        let n_values = values[0].len();
        rc_batches.push(RangeBatchInputs {
          target: RcTarget::Ab { layer: j, role },
          value_polys_fq,
          values,
          n_values,
          log_bound,
        });
      }
    }

    let (_rc_span, rc_t) = start_span!("imod_pcs_rc_shared");
    let (range_check, rc_art) =
      prove_shared_range_check(&ck.inner, &ck.eval, &rc_batches, transcript)?;
    info!(elapsed_ms = %rc_t.elapsed().as_millis(), "imod_pcs_rc_shared");

    // Route the range check's value claims to their target commitments.
    for (target, z, y) in rc_art.value_claims {
      match target {
        RcTarget::F => f_claims.push(z, y),
        RcTarget::Ab { layer, role } => {
          let mut pt = Vec::with_capacity(1 + z.len());
          pt.push(if role == 1 {
            t256::Scalar::ONE
          } else {
            t256::Scalar::ZERO
          });
          pt.extend(z);
          ab_claims[layer].push(pt, y);
        }
      }
    }

    // 8. ONE combined opening over all commitments, canonical order: `f`,
    //    the stacked layers, the chunk commitments, the multiplicity
    //    table. Interleaved tail-aligned sumchecks + a single merged
    //    same-column IPA.
    let (_bo_span, bo_t) = start_span!("imod_pcs_batched_opens");
    let mut bsub = spawn_batch_subtranscript(transcript)?;
    let mut bo_targets: Vec<(&HC, &[t256::Scalar], &HB, &OpenClaims)> =
      Vec::with_capacity(1 + t + (1 + 2 * t) + 1);
    bo_targets.push((&comm.inner, poly_fq.as_slice(), &blind.inner, &f_claims));
    for j in 0..t_layers {
      bo_targets.push((
        &ab_comms[j],
        ab_polys[j].as_slice(),
        &ab_blinds[j],
        &ab_claims[j],
      ));
    }
    for (bi, (chunk_fq, chunk_blind, claims)) in rc_art.chunk_data.iter().enumerate() {
      bo_targets.push((
        &range_check.batches[bi].chunk_comm,
        chunk_fq.as_slice(),
        chunk_blind,
        claims,
      ));
    }
    bo_targets.push((
      &range_check.mult_comm,
      rc_art.mult_fq.as_slice(),
      &rc_art.mult_blind,
      &rc_art.mult_claims,
    ));
    let combined_open = prove_combined_batch_open(&ck.inner, &ck.eval, &mut bsub, &bo_targets)?;
    info!(elapsed_ms = %bo_t.elapsed().as_millis(), "imod_pcs_batched_opens");
    info!(elapsed_ms = %prove_t.elapsed().as_millis(), "integer_modpcs_prove");

    Ok(IntEvalArgument {
      reduction_round_polys,
      int_v_prime,
      chains,
      ab_comms,
      range_check,
      combined_open,
    })
  }

  fn verify(
    vk: &Self::VerifierKey,
    transcript: &mut <T256DynPrimeEngine as SumcheckEngine>::TE,
    comm: &Self::Commitment,
    point: &[<T256DynPrimeEngine as SumcheckEngine>::Scalar],
    eval: &BigUint,
    arg: &Self::EvaluationArgument,
  ) -> Result<(), SpartanError> {
    let (_verify_span, verify_t) = start_span!("integer_modpcs_verify");
    let params = &vk.params;
    let monty = point
      .first()
      .map(|p| *p.params())
      .ok_or(SpartanError::InternalError {
        reason: "IntegerModPCS::verify: empty point".to_string(),
      })?;

    let (_vred_span, vred_t) = start_span!("imod_pcs_verify_reduction_sc");
    if arg.chains.len() != params.s {
      return Err(SpartanError::InvalidSumcheckProof);
    }
    if arg.reduction_round_polys.len() != params.numlimb_var {
      return Err(SpartanError::InvalidSumcheckProof);
    }

    // 1. Reduction sumcheck (Phase-3 step D3). Reconstruct the
    //    SumcheckProof from the round polys carried by `arg`, run
    //    `verify` with initial claim `eval` (the original Z_p eval
    //    claim) and `numlimb_var` rounds. For `numlimb_var = 0` this
    //    is a 0-round sumcheck: final_claim = eval, r_k = [].
    let eval_p = <crate::dyn_prime::DynPrime<4> as SumcheckField>::from_bytes_reduce(
      &monty,
      &eval.to_bytes_le(),
    );
    let red_sc_polys: Vec<
      crate::polys_modp::univariate::CompressedUniPoly<crate::dyn_prime::DynPrime<4>>,
    > = arg
      .reduction_round_polys
      .iter()
      .map(|coeffs| crate::polys_modp::univariate::CompressedUniPoly {
        coeffs_except_linear_term: coeffs
          .iter()
          .map(|b| {
            <crate::dyn_prime::DynPrime<4> as SumcheckField>::from_bytes_reduce(
              &monty,
              &b.to_bytes_le(),
            )
          })
          .collect(),
      })
      .collect();
    let red_sc = crate::sumcheck_modp::SumcheckProof::<T256DynPrimeEngine> {
      compressed_polys: red_sc_polys,
    };
    let (red_final_claim, r_k) =
      red_sc.verify(eval_p, params.numlimb_var, 2, &monty, transcript)?;

    // Compute limb(r_k) by MLE-evaluating the public limb weight
    // polynomial at the sumcheck challenges.
    let limb_p = build_limb_weight_dynprime(params, &monty);
    let mut limb_mle = crate::polys_modp::multilinear::MultilinearPolynomial::new(limb_p, monty);
    for r in &r_k {
      limb_mle.bind_poly_var_top(r);
    }
    let limb_at_r_k = limb_mle.into_vec()[0];
    let limb_inv = <crate::dyn_prime::DynPrime<4> as SumcheckField>::invert(&limb_at_r_k)
      .ok_or(SpartanError::InvalidSumcheckProof)?;
    let f_eval_p = red_final_claim * limb_inv;

    // 2. Check int_v' ≡ f_eval_p (mod p).
    let p = extract_p(point)?;
    let int_v_mod_p_u = arg
      .int_v_prime
      .mod_floor(&BigInt::from(p.clone()))
      .to_biguint()
      .ok_or(SpartanError::InvalidSumcheckProof)?;
    let f_eval_bu = BigUint::from_bytes_le(&f_eval_p.to_le_bytes());
    if int_v_mod_p_u != f_eval_bu {
      return Err(SpartanError::InvalidSumcheckProof);
    }

    // 3. Bind int_v' into the transcript.
    absorb_bigint(transcript, &arg.int_v_prime);

    // 4. Extend the integer point with r_k for the IntEval chain
    //    verification.
    let int_point_orig: Vec<BigUint> = point.iter().map(dyn_to_biguint).collect();
    let r_k_int: Vec<BigUint> = r_k.iter().map(dyn_to_biguint).collect();
    let int_point: Vec<BigUint> = int_point_orig
      .iter()
      .chain(r_k_int.iter())
      .cloned()
      .collect();

    let num_vars = point.len() + params.numlimb_var;
    let with_iter = num_vars > params.k;
    let n = num_vars;
    let k = params.k;
    let t = if with_iter { (n - k).div_ceil(k) } else { 0 };
    info!(elapsed_ms = %vred_t.elapsed().as_millis(), "imod_pcs_verify_reduction_sc");

    let (_vchain_span, vchain_t) = start_span!("imod_pcs_verify_chains");
    // 3. Phase 1 mirror: re-sample all `s` primes, absorb the stacked
    //    layer commitments, sample γ.
    let primes: Vec<BigUint> = (0..params.s)
      .map(|_| sample_small_prime(transcript, params.log_p))
      .collect::<Result<Vec<_>, SpartanError>>()?;
    let mut chain_primes: Vec<(BigUint, Vec<BigUint>)> = Vec::with_capacity(params.s);
    for (chain, p_i) in arg.chains.iter().zip(primes.iter()) {
      if chain.iterations.len() != t {
        return Err(SpartanError::InvalidSumcheckProof);
      }
      let r_i_int: Vec<BigUint> = int_point.iter().map(|x| x % p_i).collect();
      chain_primes.push((p_i.clone(), r_i_int));
    }
    if arg.ab_comms.len() != t {
      return Err(SpartanError::InvalidSumcheckProof);
    }
    let s_pad = params.s.next_power_of_two();
    let log_spad = s_pad.trailing_zeros() as usize;
    for c in &arg.ab_comms {
      transcript.absorb(b"ab_stacked", c);
    }

    // 4. Sample γ if iterating, identically to prover.
    let gamma_fq: Vec<t256::Scalar> = if with_iter {
      (0..(n - k))
        .map(|i| {
          let bytes = transcript.squeeze_bytes(b"gamma")?;
          let label = (i as u64).to_le_bytes();
          transcript.absorb_bytes(b"gamma_idx", &label);
          Ok(<t256::Scalar as PrimeFieldExt>::from_uniform(&bytes))
        })
        .collect::<Result<Vec<_>, SpartanError>>()?
    } else {
      Vec::new()
    };

    // 5. Per chain: re-derive the claim points, absorb the claimed evals
    //    (mirroring the prover), run the identity and CRT checks, and
    //    collect the claims for the batched-open verification.
    let shift_a_fq = biguint_to_scalar(&shift_a(params));
    let shift_b_fq = biguint_to_scalar(&shift_b(params));

    let mut f_claims = OpenClaims::default();
    let mut ab_claims: Vec<OpenClaims> = (0..t).map(|_| OpenClaims::default()).collect();
    for (chain_idx, chain) in arg.chains.iter().enumerate() {
      let (p_i, r_i_int) = &chain_primes[chain_idx];
      let p_i_fq = biguint_to_scalar(p_i);

      for (jm1, iter) in chain.iterations.iter().enumerate() {
        let j = jm1 + 1;
        let prefix_len = n - j * k;
        let lo = n - j * k;
        let hi = n - (j - 1) * k;
        let r_lower_fq: Vec<t256::Scalar> = r_i_int[lo..hi].iter().map(biguint_to_scalar).collect();
        let gamma_prefix: Vec<t256::Scalar> = gamma_fq[..prefix_len].to_vec();
        let gamma_extended: Vec<t256::Scalar> = gamma_prefix
          .iter()
          .chain(r_lower_fq.iter())
          .copied()
          .collect();

        if j == 1 {
          f_claims.push(gamma_extended.clone(), iter.a_prev_eval);
        } else {
          ab_claims[jm1 - 1].push(
            ab_stack_point(0, chain_idx, log_spad, &gamma_extended),
            iter.a_prev_eval,
          );
        }
        ab_claims[jm1].push(
          ab_stack_point(0, chain_idx, log_spad, &gamma_prefix),
          iter.a_curr_eval,
        );
        ab_claims[jm1].push(
          ab_stack_point(1, chain_idx, log_spad, &gamma_prefix),
          iter.b_curr_eval,
        );
        for ev in [&iter.a_prev_eval, &iter.a_curr_eval, &iter.b_curr_eval] {
          transcript.absorb_bytes(b"claim_ev", ev.to_repr().as_ref());
        }

        // Identity check in F: a_j(γ) + p_i · b_j(γ) ?= a_{j-1}(γ_ext).
        // a_prev for j=1 is the *unshifted* input poly (no subtract);
        // otherwise a_prev_shifted: subtract shift_a.
        let lhs_a = iter.a_curr_eval - shift_a_fq;
        let lhs_b = iter.b_curr_eval - shift_b_fq;
        let lhs = lhs_a + p_i_fq * lhs_b;
        let rhs = if j == 1 {
          iter.a_prev_eval
        } else {
          iter.a_prev_eval - shift_a_fq
        };
        if lhs != rhs {
          return Err(SpartanError::InvalidSumcheckProof);
        }
      }

      // Final-remainder claim + CRT check: (final_eval [- shift_a if t>0])
      // interpreted as a *balanced* integer ≡ int_v' (mod p_i).
      let final_point_fq: Vec<t256::Scalar> = r_i_int[..(n - t * k)]
        .iter()
        .map(biguint_to_scalar)
        .collect();
      if t == 0 {
        f_claims.push(final_point_fq, chain.final_eval);
      } else {
        ab_claims[t - 1].push(
          ab_stack_point(0, chain_idx, log_spad, &final_point_fq),
          chain.final_eval,
        );
      }
      transcript.absorb_bytes(b"claim_ev", chain.final_eval.to_repr().as_ref());

      let final_f = if t == 0 {
        chain.final_eval
      } else {
        chain.final_eval - shift_a_fq
      };
      let lhs = scalar_to_balanced_int(&final_f)
        .mod_floor(&BigInt::from(p_i.clone()))
        .to_biguint()
        .ok_or(SpartanError::InvalidSumcheckProof)?;
      let rhs = arg
        .int_v_prime
        .mod_floor(&BigInt::from(p_i.clone()))
        .to_biguint()
        .ok_or(SpartanError::InvalidSumcheckProof)?;
      if lhs != rhs {
        return Err(SpartanError::InvalidSumcheckProof);
      }
    }
    info!(elapsed_ms = %vchain_t.elapsed().as_millis(), "imod_pcs_verify_chains");

    let (_vrc_span, vrc_t) = start_span!("imod_pcs_verify_rc");
    // ONE shared LogUp-GKR range check across all (bound, size) groups,
    // in the same canonical batch order the prover used. Its value and
    // chunk claims come back for the batched-open verification.
    let log_bound_a = params.log_p + 1;
    let log_bound_b = LOG_Q - params.log_p + 1;
    let mut rc_metas: Vec<RangeBatchMeta> = Vec::with_capacity(1 + 2 * t);
    rc_metas.push(RangeBatchMeta {
      target: RcTarget::F,
      num_polys: 1,
      n_values: 1usize << num_vars,
      log_bound: params.log_t,
    });
    for j in 0..t {
      let n_values = 1usize << (num_vars - (j + 1) * params.k);
      for (role, log_bound) in [(0u8, log_bound_a), (1u8, log_bound_b)] {
        rc_metas.push(RangeBatchMeta {
          target: RcTarget::Ab { layer: j, role },
          num_polys: params.s,
          n_values,
          log_bound,
        });
      }
    }
    let rc_claims = verify_shared_range_check(&rc_metas, &arg.range_check, transcript)?;
    for (target, z, y) in rc_claims.value_claims {
      match target {
        RcTarget::F => f_claims.push(z, y),
        RcTarget::Ab { layer, role } => {
          let mut pt = Vec::with_capacity(1 + z.len());
          pt.push(if role == 1 {
            t256::Scalar::ONE
          } else {
            t256::Scalar::ZERO
          });
          pt.extend(z);
          ab_claims[layer].push(pt, y);
        }
      }
    }
    info!(elapsed_ms = %vrc_t.elapsed().as_millis(), "imod_pcs_verify_rc");

    // 6. Verify the ONE combined opening over all commitments, same
    //    canonical order as the prover.
    let (_vbo_span, vbo_t) = start_span!("imod_pcs_verify_batched_opens");
    let mut bsub = spawn_batch_subtranscript(transcript)?;
    let mut bo_targets: Vec<(
      &<Hyrax as PCSEngineTrait<T256HyraxEngine>>::Commitment,
      usize,
      &OpenClaims,
    )> = Vec::with_capacity(1 + t + (1 + 2 * t) + 1);
    bo_targets.push((&comm.inner, num_vars, &f_claims));
    for (j, ab_comm) in arg.ab_comms.iter().enumerate() {
      let m_vars = num_vars - (j + 1) * params.k;
      bo_targets.push((ab_comm, 1 + log_spad + m_vars, &ab_claims[j]));
    }
    for (bi, meta) in rc_metas.iter().enumerate() {
      let d = BatchDims::new(meta.num_polys, meta.n_values, meta.log_bound);
      bo_targets.push((
        &arg.range_check.batches[bi].chunk_comm,
        ceil_log2(d.n_chunks.max(1)),
        &rc_claims.chunk_claims[bi],
      ));
    }
    bo_targets.push((
      &arg.range_check.mult_comm,
      CHUNK_BITS,
      &rc_claims.mult_claims,
    ));
    verify_combined_batch_open(
      &vk.inner,
      &vk.eval,
      &mut bsub,
      &bo_targets,
      &arg.combined_open,
    )?;
    info!(elapsed_ms = %vbo_t.elapsed().as_millis(), "imod_pcs_verify_batched_opens");

    info!(elapsed_ms = %verify_t.elapsed().as_millis(), "integer_modpcs_verify");

    Ok(())
  }
}

/// Multilinear evaluation of `poly_fq` at point `r` over F. Mirrors the
/// dot-product form `sum_k chi(r, k) · poly[k]` used elsewhere.
fn mle_evaluate_fq(poly_fq: &[t256::Scalar], r: &[t256::Scalar]) -> t256::Scalar {
  let chis = EqPolynomial::evals_from_points(r);
  debug_assert_eq!(chis.len(), poly_fq.len());
  let mut acc = t256::Scalar::ZERO;
  for (c, v) in chis.iter().zip(poly_fq.iter()) {
    acc += *c * *v;
  }
  acc
}

/// Prover-side per-iteration state. Lives only during prove, never
/// serialized — holds the underlying F polynomial / blind / commitment
/// for both `a_j_shifted` and `b_j_shifted` so phase 2 can produce
/// openings at γ.
struct IterationProverState {
  /// `a_j_shifted` as integers; kept so the range check can re-chunk
  /// without re-shifting / re-casting from F.
  a_shifted: Vec<BigUint>,
  a_shifted_fq: Vec<t256::Scalar>,
  /// `b_j_shifted` as integers (same reason).
  b_shifted: Vec<BigUint>,
  b_shifted_fq: Vec<t256::Scalar>,
}

/// Prover-side per-chain state collected in phase 1 and consumed in
/// phase 2.
struct ChainProverState {
  p_i: BigUint,
  r_i_int: Vec<BigUint>,
  iters: Vec<IterationProverState>,
}

/// Helper: open the Hyrax commitment `comm` at `point` to produce a
/// `SmallPrimeOpening` (eval value + blind + Hyrax eval-argument). The
/// underlying polynomial `poly_fq` and its `blind` are inputs.
fn hyrax_open_at<T: ByteTranscript>(
  ck: &<Hyrax as PCSEngineTrait<T256HyraxEngine>>::CommitmentKey,
  ck_eval: &<Hyrax as PCSEngineTrait<T256HyraxEngine>>::CommitmentKey,
  transcript: &mut T,
  comm: &<Hyrax as PCSEngineTrait<T256HyraxEngine>>::Commitment,
  poly_fq: &[t256::Scalar],
  blind: &<Hyrax as PCSEngineTrait<T256HyraxEngine>>::Blind,
  point: &[t256::Scalar],
) -> Result<SmallPrimeOpening, SpartanError> {
  let f_y = mle_evaluate_fq(poly_fq, point);
  // f_y is sent in the clear in the proof, so the IPA's eval commitment
  // gains nothing from hiding. Use a zero blind and a deterministic
  // `comm_eval = G^{f_y}` that the verifier reconstructs locally; this
  // removes the per-open `Hyrax::blind` + 1-elem MSM and lets us drop
  // `blind_eval` from `SmallPrimeOpening`.
  let blind_eval = HyraxBlind::<T256HyraxEngine>::zero(ck_eval, 1);
  let comm_eval = Hyrax::commit(ck_eval, &[f_y], &blind_eval, false)?;
  let arg = Hyrax::prove(
    ck,
    ck_eval,
    transcript,
    comm,
    poly_fq,
    blind,
    point,
    &comm_eval,
    &blind_eval,
  )?;
  Ok(SmallPrimeOpening {
    f_y,
    hyrax_arg: arg,
  })
}

/// Mirror of `hyrax_open_at` on the verifier side: reconstruct
/// `comm_eval = G^{f_y}` (zero blind) and verify the Hyrax argument
/// against the polynomial commitment `comm` at `point`.
fn hyrax_verify_open<T: ByteTranscript>(
  vk: &<Hyrax as PCSEngineTrait<T256HyraxEngine>>::VerifierKey,
  ck_eval: &<Hyrax as PCSEngineTrait<T256HyraxEngine>>::CommitmentKey,
  transcript: &mut T,
  comm: &<Hyrax as PCSEngineTrait<T256HyraxEngine>>::Commitment,
  point: &[t256::Scalar],
  opening: &SmallPrimeOpening,
) -> Result<(), SpartanError> {
  let zero_blind = HyraxBlind::<T256HyraxEngine>::zero(ck_eval, 1);
  let comm_eval = Hyrax::commit(ck_eval, &[opening.f_y], &zero_blind, false)?;
  Hyrax::verify(
    vk,
    ck_eval,
    transcript,
    comm,
    point,
    &comm_eval,
    &opening.hyrax_arg,
  )
}

/// Which commitment a range-check batch's value claim targets.
#[derive(Clone, Copy, Debug)]
enum RcTarget {
  /// The input polynomial `f` (the `f_limb` batch).
  F,
  /// The stacked `(role, chain, x)` commitment of iteration layer `layer`
  /// (0-indexed), restricted to `role` (0 = a, 1 = b).
  Ab { layer: usize, role: u8 },
}

/// Accumulated multi-point evaluation claims against one commitment.
#[derive(Default, Clone, Debug)]
struct OpenClaims {
  points: Vec<Vec<t256::Scalar>>,
  evals: Vec<t256::Scalar>,
}

impl OpenClaims {
  fn push(&mut self, point: Vec<t256::Scalar>, eval: t256::Scalar) {
    self.points.push(point);
    self.evals.push(eval);
  }
}

/// Inputs for one homogeneous batch of the shared range check: `N` value
/// polynomials, all of length `n_values` (a power of two) and the same
/// bound `2^log_bound`. The value polynomials are NOT separately
/// committed — their `V(r_v)` evaluation becomes a claim on `target`.
struct RangeBatchInputs<'a> {
  target: RcTarget,
  value_polys_fq: Vec<&'a [t256::Scalar]>,
  values: Vec<&'a [BigUint]>,
  /// Coefficients per polynomial (same for all; a power of two).
  n_values: usize,
  /// Bit-width of the shared bound (each value `< 2^log_bound`).
  log_bound: usize,
}

/// Verifier-side metadata for one batch of the shared range check.
struct RangeBatchMeta {
  target: RcTarget,
  num_polys: usize,
  n_values: usize,
  log_bound: usize,
}

/// Sizes derived from a batch's public parameters, shared by prover and
/// verifier.
#[derive(Clone, Copy)]
struct BatchDims {
  log_np: usize,
  log_nv: usize,
  numchunks: usize,
  stride: usize,
  log_stride: usize,
  n_chunks: usize,
  /// Bit-width of the top chunk, `log_bound − 16·(numchunks−1)` ∈ [1, 16].
  rem: usize,
}

impl BatchDims {
  fn new(num_polys: usize, n_values: usize, log_bound: usize) -> Self {
    let n_pad = num_polys.next_power_of_two();
    let log_np = n_pad.trailing_zeros() as usize;
    let log_nv = ceil_log2(n_values.max(1));
    let numchunks = log_bound.div_ceil(CHUNK_BITS);
    // Min stride 2 keeps the reconstruction sumcheck non-degenerate when a
    // bound fits in a single chunk (the extra slot is zero-valued and
    // zero-weighted).
    let stride = numchunks.next_power_of_two().max(2);
    Self {
      log_np,
      log_nv,
      numchunks,
      stride,
      log_stride: stride.trailing_zeros() as usize,
      n_chunks: n_pad * n_values * stride,
      rem: log_bound - CHUNK_BITS * (numchunks - 1),
    }
  }

  /// Whether the top chunk needs the shifted-lookup tightening.
  fn top_needed(&self) -> bool {
    self.rem < CHUNK_BITS
  }

  /// The public shift `2^16 − 2^rem` applied to top chunks so the same
  /// `2^16` table enforces `top < 2^rem`.
  fn top_shift(&self) -> u64 {
    (1u64 << CHUNK_BITS) - (1u64 << self.rem)
  }
}

/// Masked base-`2^16` weight vector for the value-reconstruction
/// sumcheck: `w[c] = 2^(16c)` for `c < ⌈log_bound/16⌉`, else `0`. Length
/// `stride` (the padded per-value chunk count). Chunk slots at
/// `c ≥ numchunks` carry zero weight, so the prover can't inflate a
/// value past its bound regardless of those (still range-checked) slots.
fn chunk_weight_vector(log_bound: usize, stride: usize) -> Vec<t256::Scalar> {
  let numchunks = log_bound.div_ceil(CHUNK_BITS);
  let base = t256::Scalar::from(1u64 << CHUNK_BITS);
  let mut weight = Vec::with_capacity(stride);
  let mut pow = t256::Scalar::ONE;
  for c in 0..stride {
    if c < numchunks {
      weight.push(pow);
      pow *= base;
    } else {
      weight.push(t256::Scalar::ZERO);
    }
  }
  weight
}

/// Spawn the F-side sub-transcript of the shared range check, seeded
/// from the parent and binding every batch's chunk commitment plus the
/// shared multiplicity commitment — all before any challenge (in
/// particular the LogUp `r`) is squeezed. Both prover and verifier
/// reconstruct it identically.
fn spawn_shared_range_subtranscript<'a>(
  parent: &mut Keccak256Transcript<T256DynPrimeEngine>,
  chunk_comms: impl Iterator<Item = &'a <Hyrax as PCSEngineTrait<T256HyraxEngine>>::Commitment>,
  mult_comm: &<Hyrax as PCSEngineTrait<T256HyraxEngine>>::Commitment,
) -> Result<Keccak256Transcript<T256HyraxEngine>, SpartanError> {
  let seed = parent.squeeze_bytes(b"range_seed")?;
  let mut sub = <Keccak256Transcript<T256HyraxEngine> as TranscriptEngineTrait<
    T256HyraxEngine,
  >>::new_with_params(b"range_check", ());
  sub.absorb_bytes(b"seed", &seed);
  for cc in chunk_comms {
    sub.absorb(b"range_chunk_comm", cc);
  }
  sub.absorb(b"range_mult_comm", mult_comm);
  Ok(sub)
}

/// Spawn the F-side sub-transcript of the final batched-open phase.
/// Each commitment's claims are absorbed (and its λ squeezed) inside
/// [`prove_batched_open`] / [`verify_batched_open`].
fn spawn_batch_subtranscript(
  parent: &mut Keccak256Transcript<T256DynPrimeEngine>,
) -> Result<Keccak256Transcript<T256HyraxEngine>, SpartanError> {
  let seed = parent.squeeze_bytes(b"batch_seed")?;
  let mut sub = <Keccak256Transcript<T256HyraxEngine> as TranscriptEngineTrait<
    T256HyraxEngine,
  >>::new_with_params(b"batched_opens", ());
  sub.absorb_bytes(b"seed", &seed);
  Ok(sub)
}

/// Boolean-extended point selecting the `(role, chain)` block of a
/// stacked per-layer `(a, b)` polynomial: `[role] ++ bits(chain) ++
/// sub_point`.
fn ab_stack_point(
  role: u8,
  chain: usize,
  log_spad: usize,
  sub_point: &[t256::Scalar],
) -> Vec<t256::Scalar> {
  let mut pt = Vec::with_capacity(1 + log_spad + sub_point.len());
  pt.push(if role == 1 {
    t256::Scalar::ONE
  } else {
    t256::Scalar::ZERO
  });
  pt.extend(bool_point_of_index(chain, log_spad));
  pt.extend_from_slice(sub_point);
  pt
}

/// Absorb a commitment and its claims into the batch sub-transcript,
/// binding them before the RLC challenge λ is squeezed.
fn absorb_batch_claims(
  sub: &mut Keccak256Transcript<T256HyraxEngine>,
  comm: &<Hyrax as PCSEngineTrait<T256HyraxEngine>>::Commitment,
  claims: &OpenClaims,
) {
  sub.absorb(b"bo_comm", comm);
  for (z, y) in claims.points.iter().zip(claims.evals.iter()) {
    for c in z {
      sub.absorb_bytes(b"bo_pt", c.to_repr().as_ref());
    }
    sub.absorb_bytes(b"bo_ev", y.to_repr().as_ref());
  }
}

/// Prove the combined multi-point opening (see [`CombinedBatchOpen`]).
/// `targets` are `(commitment, poly, blind, claims)` in canonical order.
fn prove_combined_batch_open(
  ck: &<Hyrax as PCSEngineTrait<T256HyraxEngine>>::CommitmentKey,
  ck_eval: &<Hyrax as PCSEngineTrait<T256HyraxEngine>>::CommitmentKey,
  sub: &mut Keccak256Transcript<T256HyraxEngine>,
  targets: &[(
    &<Hyrax as PCSEngineTrait<T256HyraxEngine>>::Commitment,
    &[t256::Scalar],
    &<Hyrax as PCSEngineTrait<T256HyraxEngine>>::Blind,
    &OpenClaims,
  )],
) -> Result<CombinedBatchOpen, SpartanError> {
  use crate::polys::univariate::UniPoly;
  let m = targets.len();

  // 1. Per commitment: bind claims, squeeze λ, build W and the combined
  //    running claim.
  let (_bw_span, bw_t) = start_span!("bo_w_build");
  let mut fs = Vec::with_capacity(m);
  let mut ws = Vec::with_capacity(m);
  let mut run = Vec::with_capacity(m);
  let mut nv = Vec::with_capacity(m);
  for (comm, poly, _, claims) in targets {
    debug_assert!(!claims.points.is_empty());
    absorb_batch_claims(sub, comm, claims);
    let lambda = sub.squeeze(b"bo_lambda")?;
    let (w, c) = batch_weight(&claims.points, &claims.evals, lambda, poly.len());
    fs.push(crate::polys::multilinear::MultilinearPolynomial::new(
      poly.to_vec(),
    ));
    ws.push(crate::polys::multilinear::MultilinearPolynomial::new(w));
    run.push(c);
    nv.push(poly.len().trailing_zeros() as usize);
  }
  let n_max = *nv.iter().max().expect("non-empty targets");
  info!(elapsed_ms = %bw_t.elapsed().as_millis(), m = %m, "bo_w_build");
  let (_bs_span, bs_t) = start_span!("bo_interleaved_sc");

  // 2. Interleaved, tail-aligned sumcheck rounds: per global round, all
  //    active instances absorb their round polynomial, then ONE shared
  //    challenge is squeezed and binds them all.
  let mut round_polys: Vec<Vec<crate::polys::univariate::CompressedUniPoly<t256::Scalar>>> =
    vec![Vec::new(); m];
  let mut challenges: Vec<t256::Scalar> = Vec::with_capacity(n_max);
  for g in 0..n_max {
    let mut staged: Vec<(usize, UniPoly<t256::Scalar>)> = Vec::new();
    for j in 0..m {
      if g < n_max - nv[j] {
        continue; // joins later
      }
      let len = fs[j].Z.len();
      let h = len / 2;
      let (e0, e2) = if h >= 1 << 12 {
        (0..h)
          .into_par_iter()
          .map(|i| {
            let (f0, f1) = (fs[j].Z[i], fs[j].Z[i + h]);
            let (w0, w1) = (ws[j].Z[i], ws[j].Z[i + h]);
            (f0 * w0, (f1 + f1 - f0) * (w1 + w1 - w0))
          })
          .reduce(
            || (t256::Scalar::ZERO, t256::Scalar::ZERO),
            |a, b| (a.0 + b.0, a.1 + b.1),
          )
      } else {
        let mut e0 = t256::Scalar::ZERO;
        let mut e2 = t256::Scalar::ZERO;
        for i in 0..h {
          let (f0, f1) = (fs[j].Z[i], fs[j].Z[i + h]);
          let (w0, w1) = (ws[j].Z[i], ws[j].Z[i + h]);
          e0 += f0 * w0;
          e2 += (f1 + f1 - f0) * (w1 + w1 - w0);
        }
        (e0, e2)
      };
      let uni = UniPoly::from_evals(&[e0, run[j] - e0, e2])?;
      sub.absorb(b"cbo_p", &uni);
      staged.push((j, uni));
    }
    let r_g = sub.squeeze(b"cbo_c")?;
    for (j, uni) in staged {
      run[j] = uni.evaluate(&r_g);
      round_polys[j].push(uni.compress());
      fs[j].bind_poly_var_top(&r_g);
      ws[j].bind_poly_var_top(&r_g);
    }
    challenges.push(r_g);
  }

  info!(elapsed_ms = %bs_t.elapsed().as_millis(), "bo_interleaved_sc");

  // 3. Send the per-commitment final evaluations, then μ.
  let final_evals: Vec<t256::Scalar> = fs.iter().map(|f| f[0]).collect();
  for y in &final_evals {
    sub.absorb_bytes(b"cbo_fe", y.to_repr().as_ref());
  }
  let mu_bytes = sub.squeeze_bytes(b"cbo_mu")?;
  let mu = <t256::Scalar as PrimeFieldExt>::from_uniform(&mu_bytes);

  let (_bm_span, bm_t) = start_span!("bo_merged_ipa");
  // 4. Merge every column-width-or-larger commitment into ONE same-column
  //    IPA; open the rest (test sizes) individually.
  let width = Hyrax::key_num_cols(ck);
  let mut small_opens = Vec::new();
  let mut big_items: Vec<(
    &<Hyrax as PCSEngineTrait<T256HyraxEngine>>::Commitment,
    &[t256::Scalar],
    &<Hyrax as PCSEngineTrait<T256HyraxEngine>>::Blind,
    Vec<t256::Scalar>,
  )> = Vec::new();
  let mut y_star = t256::Scalar::ZERO;
  let mut mu_pow = t256::Scalar::ONE;
  for (j, (comm, poly, blind, _)) in targets.iter().enumerate() {
    let r_j = challenges[n_max - nv[j]..].to_vec();
    if poly.len() >= width {
      y_star += mu_pow * final_evals[j];
      mu_pow *= mu;
      big_items.push((comm, poly, blind, r_j));
    } else {
      small_opens.push(hyrax_open_at(ck, ck_eval, sub, comm, poly, blind, &r_j)?);
    }
  }
  let merged = if big_items.is_empty() {
    None
  } else {
    // `y_star` is recomputed by the verifier from the in-clear
    // `final_evals`, so the eval commitment needs no hiding: use the
    // deterministic zero-blind `G^{y_star}` (same treatment as
    // `hyrax_open_at`).
    let blind_eval = HyraxBlind::<T256HyraxEngine>::zero(ck_eval, 1);
    let comm_eval = Hyrax::commit(ck_eval, &[y_star], &blind_eval, false)?;
    let arg =
      Hyrax::prove_same_column_batch(ck, ck_eval, sub, &big_items, mu, &comm_eval, &blind_eval)?;
    Some(SmallPrimeOpening {
      f_y: y_star,
      hyrax_arg: arg,
    })
  };

  info!(elapsed_ms = %bm_t.elapsed().as_millis(), "bo_merged_ipa");

  Ok(CombinedBatchOpen {
    round_polys,
    final_evals,
    merged,
    small_opens,
  })
}

/// Verifier mirror of [`prove_combined_batch_open`]. `targets` are
/// `(commitment, num_vars, claims)` in canonical order; every claim's
/// point length is pinned to its commitment's variable count.
fn verify_combined_batch_open(
  vk: &<Hyrax as PCSEngineTrait<T256HyraxEngine>>::VerifierKey,
  ck_eval: &<Hyrax as PCSEngineTrait<T256HyraxEngine>>::CommitmentKey,
  sub: &mut Keccak256Transcript<T256HyraxEngine>,
  targets: &[(
    &<Hyrax as PCSEngineTrait<T256HyraxEngine>>::Commitment,
    usize,
    &OpenClaims,
  )],
  arg: &CombinedBatchOpen,
) -> Result<(), SpartanError> {
  let m = targets.len();
  if arg.round_polys.len() != m || arg.final_evals.len() != m {
    return Err(SpartanError::InvalidSumcheckProof);
  }
  let mut lambdas = Vec::with_capacity(m);
  let mut run = Vec::with_capacity(m);
  let mut nv = Vec::with_capacity(m);
  for (j, (comm, num_vars, claims)) in targets.iter().enumerate() {
    if claims.points.is_empty() || claims.points.iter().any(|z| z.len() != *num_vars) {
      return Err(SpartanError::InvalidSumcheckProof);
    }
    if arg.round_polys[j].len() != *num_vars {
      return Err(SpartanError::InvalidSumcheckProof);
    }
    absorb_batch_claims(sub, comm, claims);
    let lambda = sub.squeeze(b"bo_lambda")?;
    let mut c = t256::Scalar::ZERO;
    let mut lam_pow = t256::Scalar::ONE;
    for y in &claims.evals {
      c += lam_pow * y;
      lam_pow *= lambda;
    }
    lambdas.push(lambda);
    run.push(c);
    nv.push(*num_vars);
  }
  let n_max = *nv.iter().max().expect("non-empty targets");

  let mut challenges: Vec<t256::Scalar> = Vec::with_capacity(n_max);
  for g in 0..n_max {
    let mut staged: Vec<(usize, crate::polys::univariate::UniPoly<t256::Scalar>)> = Vec::new();
    for j in 0..m {
      if g < n_max - nv[j] {
        continue;
      }
      let uni = arg.round_polys[j][g - (n_max - nv[j])].decompress(&run[j]);
      if uni.degree() != 2 {
        return Err(SpartanError::InvalidSumcheckProof);
      }
      sub.absorb(b"cbo_p", &uni);
      staged.push((j, uni));
    }
    let r_g = sub.squeeze(b"cbo_c")?;
    for (j, uni) in staged {
      run[j] = uni.evaluate(&r_g);
    }
    challenges.push(r_g);
  }

  // Final sumcheck checks: run_j == f_j(r_j)·W_j(r_j), with W_j(r_j)
  // recomputed in closed form.
  for (j, (_, _, claims)) in targets.iter().enumerate() {
    let r_j = &challenges[n_max - nv[j]..];
    let mut w_at_r = t256::Scalar::ZERO;
    let mut lam_pow = t256::Scalar::ONE;
    for z in &claims.points {
      w_at_r += lam_pow * EqPolynomial::<t256::Scalar>::new(z.clone()).evaluate(r_j);
      lam_pow *= lambdas[j];
    }
    if run[j] != arg.final_evals[j] * w_at_r {
      return Err(SpartanError::InvalidSumcheckProof);
    }
  }

  for y in &arg.final_evals {
    sub.absorb_bytes(b"cbo_fe", y.to_repr().as_ref());
  }
  let mu_bytes = sub.squeeze_bytes(b"cbo_mu")?;
  let mu = <t256::Scalar as PrimeFieldExt>::from_uniform(&mu_bytes);

  let width = Hyrax::vk_num_cols(vk);
  let mut small_iter = arg.small_opens.iter();
  let mut big_items: Vec<(
    &<Hyrax as PCSEngineTrait<T256HyraxEngine>>::Commitment,
    Vec<t256::Scalar>,
  )> = Vec::new();
  let mut y_star = t256::Scalar::ZERO;
  let mut mu_pow = t256::Scalar::ONE;
  for (j, (comm, num_vars, _)) in targets.iter().enumerate() {
    let r_j = challenges[n_max - nv[j]..].to_vec();
    if (1usize << num_vars) >= width {
      y_star += mu_pow * arg.final_evals[j];
      mu_pow *= mu;
      big_items.push((comm, r_j));
    } else {
      let open = small_iter
        .next()
        .ok_or(SpartanError::InvalidSumcheckProof)?;
      hyrax_verify_open(vk, ck_eval, sub, comm, &r_j, open)?;
      if open.f_y != arg.final_evals[j] {
        return Err(SpartanError::InvalidSumcheckProof);
      }
    }
  }
  if small_iter.next().is_some() {
    return Err(SpartanError::InvalidSumcheckProof);
  }
  if big_items.is_empty() {
    if arg.merged.is_some() {
      return Err(SpartanError::InvalidSumcheckProof);
    }
  } else {
    let merged = arg
      .merged
      .as_ref()
      .ok_or(SpartanError::InvalidSumcheckProof)?;
    if merged.f_y != y_star {
      return Err(SpartanError::InvalidSumcheckProof);
    }
    let zero_blind = HyraxBlind::<T256HyraxEngine>::zero(ck_eval, 1);
    let comm_eval = Hyrax::commit(ck_eval, &[merged.f_y], &zero_blind, false)?;
    Hyrax::verify_same_column_batch(
      vk,
      ck_eval,
      sub,
      &big_items,
      mu,
      &comm_eval,
      &merged.hyrax_arg,
    )?;
  }
  Ok(())
}

/// Build `W = Σ_i λ^i·eq(z_i, ·)` (length `n`) and the combined claim
/// `Σ_i λ^i·y_i` for a batched open, exploiting claim-point structure so
/// the cost is far below the naive `#claims · n`:
///
/// - **Boolean head**: a claim whose leading coordinates are exactly
///   `0/1` (stacked-layer points `(role, bits(chain), ·)`, role-prefixed
///   range-check points) has `eq(z, ·)` supported on a single block —
///   write `λ^i·eq(tail)` into that block, cost `2^(n_vars − h)`.
/// - **Shared random prefix**: consecutive claims sharing a common
///   prefix (the `j=1` a_prev points `(γ_prefix, r^(c))`) are
///   tensor-combined: `eq(prefix) ⊗ Σ_c λ^c·eq(tail_c)` — one full-size
///   pass per *group* instead of per claim.
///
/// The output is bit-identical to the naive construction (the structure
/// only changes how the same table is assembled), so the transcript and
/// verifier are unaffected.
fn batch_weight(
  points: &[Vec<t256::Scalar>],
  evals: &[t256::Scalar],
  lambda: t256::Scalar,
  n: usize,
) -> (Vec<t256::Scalar>, t256::Scalar) {
  let n_vars = n.trailing_zeros() as usize;
  debug_assert_eq!(1usize << n_vars, n);
  let mut w = vec![t256::Scalar::ZERO; n];
  let mut claim = t256::Scalar::ZERO;
  let mut lam_pow = t256::Scalar::ONE;

  // Length of the leading run of exactly-boolean coordinates.
  let bool_head = |z: &[t256::Scalar]| -> usize {
    z.iter()
      .take_while(|c| **c == t256::Scalar::ZERO || **c == t256::Scalar::ONE)
      .count()
  };
  // Longest common prefix of two points.
  let lcp = |a: &[t256::Scalar], b: &[t256::Scalar]| -> usize {
    a.iter().zip(b.iter()).take_while(|(x, y)| x == y).count()
  };

  let mut i = 0;
  while i < points.len() {
    let z = &points[i];
    debug_assert_eq!(z.len(), n_vars);
    let h = bool_head(z);
    if h > 0 {
      // Block write: λ^i·eq(z[h..]) into the block selected by z[..h].
      let mut block = 0usize;
      for c in &z[..h] {
        block = (block << 1) | usize::from(*c == t256::Scalar::ONE);
      }
      let tail = EqPolynomial::<t256::Scalar>::evals_from_points(&z[h..]);
      let bs = tail.len();
      let lam = lam_pow;
      w[block * bs..(block + 1) * bs]
        .par_iter_mut()
        .zip(tail.par_iter())
        .for_each(|(wj, e)| *wj += lam * e);
      claim += lam_pow * evals[i];
      lam_pow *= lambda;
      i += 1;
      continue;
    }

    // Group consecutive claims sharing a (random) prefix with z. Distinct
    // protocol points never collide by accident over a 256-bit field, so
    // any LCP > 0 identifies genuinely shared structure.
    let mut j = i + 1;
    let mut p = n_vars;
    while j < points.len() && lcp(z, &points[j]) > 0 {
      p = p.min(lcp(z, &points[j]));
      j += 1;
    }
    if j > i + 1 && p > 0 && p < n_vars {
      // Tensor: eq(z[..p]) ⊗ Σ_c λ^c·eq(z_c[p..]).
      let tail_len = 1usize << (n_vars - p);
      let mut small = vec![t256::Scalar::ZERO; tail_len];
      for c in i..j {
        debug_assert_eq!(&points[c][..p], &z[..p]);
        let tail = EqPolynomial::<t256::Scalar>::evals_from_points(&points[c][p..]);
        let lam = lam_pow;
        for (sj, e) in small.iter_mut().zip(tail.iter()) {
          *sj += lam * e;
        }
        claim += lam_pow * evals[c];
        lam_pow *= lambda;
      }
      let pref = EqPolynomial::<t256::Scalar>::evals_from_points(&z[..p]);
      w.par_chunks_mut(tail_len)
        .zip(pref.par_iter())
        .for_each(|(chunk, pe)| {
          for (wj, sj) in chunk.iter_mut().zip(small.iter()) {
            *wj += *pe * sj;
          }
        });
      i = j;
      continue;
    }

    // Singleton: one full-size accumulate.
    let eq_c = EqPolynomial::<t256::Scalar>::evals_from_points(z);
    let lam = lam_pow;
    w.par_iter_mut()
      .zip(eq_c.par_iter())
      .for_each(|(wj, e)| *wj += lam * e);
    claim += lam_pow * evals[i];
    lam_pow *= lambda;
    i += 1;
  }
  (w, claim)
}

/// Prover-side artifacts of the shared range check that feed the final
/// batched opens: value claims routed to `f` / stacked-layer targets,
/// per-batch chunk polynomials + blinds + claims, and the multiplicity
/// table's polynomial + blind + claim.
struct RcProverArtifacts {
  value_claims: Vec<(RcTarget, Vec<t256::Scalar>, t256::Scalar)>,
  chunk_data: Vec<(
    Vec<t256::Scalar>,
    <Hyrax as PCSEngineTrait<T256HyraxEngine>>::Blind,
    OpenClaims,
  )>,
  mult_fq: Vec<t256::Scalar>,
  mult_blind: <Hyrax as PCSEngineTrait<T256HyraxEngine>>::Blind,
  mult_claims: OpenClaims,
}

/// Claims the verifier of the shared range check hands back for the
/// batched-open verification.
struct RcVerifyClaims {
  value_claims: Vec<(RcTarget, Vec<t256::Scalar>, t256::Scalar)>,
  chunk_claims: Vec<OpenClaims>,
  mult_claims: OpenClaims,
}

/// Prover side of the shared LogUp-GKR range check covering all batches
/// of one Mod-PCS opening. Per batch: build and commit the stacked chunk
/// polynomial. Shared: one multiplicity table and one multi-witness
/// LogUp whose witness trees are all batches' chunk polys plus the
/// shifted-top-chunk sub-polys of non-16-aligned batches. All
/// evaluation obligations (LogUp witness/table claims, top claims, the
/// per-batch `V(r_v)` value claims, and the reconstruction sumchecks'
/// final chunk evaluations) are returned as CLAIMS to be discharged by
/// the caller's batched opens — this function performs no Hyrax opens.
fn prove_shared_range_check(
  ck: &<Hyrax as PCSEngineTrait<T256HyraxEngine>>::CommitmentKey,
  _ck_eval: &<Hyrax as PCSEngineTrait<T256HyraxEngine>>::CommitmentKey,
  batches: &[RangeBatchInputs<'_>],
  parent: &mut Keccak256Transcript<T256DynPrimeEngine>,
) -> Result<(SharedRangeCheck, RcProverArtifacts), SpartanError> {
  type HC = <Hyrax as PCSEngineTrait<T256HyraxEngine>>::Commitment;
  type HB = <Hyrax as PCSEngineTrait<T256HyraxEngine>>::Blind;
  debug_assert!(!batches.is_empty());

  let dims: Vec<BatchDims> = batches
    .iter()
    .map(|b| BatchDims::new(b.value_polys_fq.len(), b.n_values, b.log_bound))
    .collect();

  // 1. Per batch: stacked chunk polynomial (u64 entries, each < 2^16).
  //    Index `((p·n_values + within)·stride + c)`. Padding polys
  //    (`p ≥ num_polys`) and slots `c ≥ numchunks` stay zero (zero is in
  //    the table, and those slots carry zero weight).
  let (_rcc_span, rcc_t) = start_span!("rc_chunk_commit");
  let mut chunk_vals_all: Vec<Vec<u64>> = Vec::with_capacity(batches.len());
  let mut chunk_fq_all: Vec<Vec<t256::Scalar>> = Vec::with_capacity(batches.len());
  let mut chunk_blinds: Vec<HB> = Vec::with_capacity(batches.len());
  let mut chunk_comms: Vec<HC> = Vec::with_capacity(batches.len());
  for (b, d) in batches.iter().zip(dims.iter()) {
    let num_polys = b.value_polys_fq.len();
    debug_assert!(num_polys >= 1);
    debug_assert!(b.n_values.is_power_of_two());
    debug_assert!(b.values.iter().all(|v| v.len() == b.n_values));
    info!(
      num_polys = num_polys,
      n_values = b.n_values,
      log_bound = b.log_bound,
      stride = d.stride,
      n_chunks = d.n_chunks,
      "imod_pcs_range_batch"
    );
    let mut chunk_vals: Vec<u64> = vec![0u64; d.n_chunks];
    chunk_vals
      .par_chunks_mut(d.stride)
      .enumerate()
      .for_each(|(gv, slot)| {
        let p = gv / b.n_values;
        if p >= num_polys {
          return; // padding poly: all-zero
        }
        let within = gv % b.n_values;
        for (c, ch) in chunk_decompose_value(&b.values[p][within], b.log_bound)
          .into_iter()
          .enumerate()
        {
          slot[c] = ch;
        }
      });
    let chunk_fq: Vec<t256::Scalar> = chunk_vals
      .par_iter()
      .map(|&c| t256::Scalar::from(c))
      .collect();
    let blind = Hyrax::blind(ck, d.n_chunks);
    let comm = Hyrax::commit(ck, &chunk_fq, &blind, true)?;
    chunk_vals_all.push(chunk_vals);
    chunk_fq_all.push(chunk_fq);
    chunk_blinds.push(blind);
    chunk_comms.push(comm);
  }

  info!(elapsed_ms = %rcc_t.elapsed().as_millis(), "rc_chunk_commit");

  // 2. Shifted top chunks of the non-16-aligned batches: `top + (2^16 −
  //    2^rem)` is in the 2^16 table iff `top < 2^rem`. These become extra
  //    LogUp witness trees; no extra commitment (their MLE is the chunk
  //    MLE at a boolean-extended point, plus the public shift).
  let mut top_vals_all: Vec<(usize, Vec<u64>)> = Vec::new(); // (batch idx, shifted tops)
  for (bi, d) in dims.iter().enumerate() {
    if d.top_needed() {
      let shift = d.top_shift();
      let stride = d.stride;
      let tops: Vec<u64> = (0..d.n_chunks / stride)
        .map(|gv| chunk_vals_all[bi][gv * stride + (d.numchunks - 1)] + shift)
        .collect();
      top_vals_all.push((bi, tops));
    }
  }

  // 3. The shared multiplicity table over ALL witness trees, committed
  //    before the LogUp challenge `r` is squeezed (multiplicities chosen
  //    after `r` would break the lookup identity).
  let witness_refs: Vec<&[u64]> = chunk_vals_all
    .iter()
    .map(|v| v.as_slice())
    .chain(top_vals_all.iter().map(|(_, v)| v.as_slice()))
    .collect();
  let (_rcm_span, rcm_t) = start_span!("rc_mult_commit");
  let mult = crate::logup_gkr::LogUpMultiRangeProof::<T256HyraxEngine>::multiplicities(
    CHUNK_BITS,
    &witness_refs,
  )?;
  let mult_fq: Vec<t256::Scalar> = mult.iter().map(|&m| t256::Scalar::from(m)).collect();
  let mult_blind = Hyrax::blind(ck, mult_fq.len());
  let mult_comm = Hyrax::commit(ck, &mult_fq, &mult_blind, true)?;
  info!(elapsed_ms = %rcm_t.elapsed().as_millis(), "rc_mult_commit");

  // 4. Sub-transcript bound to (parent, chunk comms, mult comm).
  let mut sub = spawn_shared_range_subtranscript(parent, chunk_comms.iter(), &mult_comm)?;

  // 5. ONE multi-witness LogUp-GKR: every entry of every tree is in
  //    [0, 2^16). Its reduced claims become batched-open claims.
  let (_rcl_span, rcl_t) = start_span!("rc_logup_gkr");
  let (logup, claims) = crate::logup_gkr::LogUpMultiRangeProof::<T256HyraxEngine>::prove(
    CHUNK_BITS,
    &chunk_vals_all
      .iter()
      .map(|v| v.as_slice())
      .chain(top_vals_all.iter().map(|(_, v)| v.as_slice()))
      .collect::<Vec<_>>(),
    &mut sub,
  )?;
  info!(elapsed_ms = %rcl_t.elapsed().as_millis(), "rc_logup_gkr");
  let mut chunk_claims: Vec<OpenClaims> = vec![OpenClaims::default(); batches.len()];
  for (bi, (point, eval)) in claims.wit_claims.iter().take(batches.len()).enumerate() {
    chunk_claims[bi].push(point.clone(), *eval);
  }
  for (ti, (bi, _)) in top_vals_all.iter().enumerate() {
    let d = &dims[*bi];
    let (point, eval) = &claims.wit_claims[batches.len() + ti];
    let ext: Vec<t256::Scalar> = point
      .iter()
      .copied()
      .chain(bool_point_of_index(d.numchunks - 1, d.log_stride))
      .collect();
    chunk_claims[*bi].push(ext, *eval - t256::Scalar::from(d.top_shift()));
  }
  let mut mult_claims = OpenClaims::default();
  mult_claims.push(claims.mult_point.clone(), claims.mult_eval);

  // 6. Per batch: value-reconstruction sumcheck tying the chunks to the
  //    batch's value polynomials. The folded value `V(r_v) =
  //    Σ_p eq(r_v_poly, p)·value_p(r_v_within)` is sent as a claim on the
  //    batch's target commitment, and the sumcheck's final chunk
  //    evaluation is a claim on the chunk commitment.
  let (_rcr_span, rcr_t) = start_span!("rc_reconstr");
  let mut value_claims: Vec<(RcTarget, Vec<t256::Scalar>, t256::Scalar)> =
    Vec::with_capacity(batches.len());
  let mut batch_data: Vec<RangeCheckBatchData> = Vec::with_capacity(batches.len());
  for (bi, (b, d)) in batches.iter().zip(dims.iter()).enumerate() {
    let num_polys = b.value_polys_fq.len();
    let r_v: Vec<t256::Scalar> = (0..(d.log_np + d.log_nv))
      .map(|_| sub.squeeze(b"range_rv"))
      .collect::<Result<Vec<_>, _>>()?;
    let r_v_poly = &r_v[..d.log_np];
    let r_v_within = &r_v[d.log_np..];

    let eq_weights = EqPolynomial::<t256::Scalar>::new(r_v_poly.to_vec()).evals();
    let w = &eq_weights[..num_polys];
    let mut combined_poly = vec![t256::Scalar::ZERO; b.n_values];
    for (p, poly) in b.value_polys_fq.iter().enumerate() {
      for (o, &v) in combined_poly.iter_mut().zip(poly.iter()) {
        *o += w[p] * v;
      }
    }
    let value_eval = mle_evaluate_fq(&combined_poly, r_v_within);
    sub.absorb_bytes(b"rc_value_ev", value_eval.to_repr().as_ref());
    value_claims.push((b.target, r_v.clone(), value_eval));

    // Partial-eval chunk poly at r_v, leaving the chunk axis.
    let mut chunk_mle =
      crate::polys::multilinear::MultilinearPolynomial::new(chunk_fq_all[bi].clone());
    for r in &r_v {
      chunk_mle.bind_poly_var_top(r);
    }
    let chunk_at_rv: Vec<t256::Scalar> = chunk_mle.into_vec();
    debug_assert_eq!(chunk_at_rv.len(), d.stride);

    let weight = chunk_weight_vector(b.log_bound, d.stride);
    let mut poly_w = crate::polys::multilinear::MultilinearPolynomial::new(weight);
    let mut poly_c = crate::polys::multilinear::MultilinearPolynomial::new(chunk_at_rv);
    let (value_reconstr_sumcheck, r_b, sc_claims) =
      crate::sumcheck::SumcheckProof::<T256HyraxEngine>::prove_quad(
        &value_eval,
        d.log_stride,
        &mut poly_w,
        &mut poly_c,
        &mut sub,
      )?;
    let reconstr_eval = sc_claims[1];
    sub.absorb_bytes(b"rc_reconstr_ev", reconstr_eval.to_repr().as_ref());
    let combined: Vec<t256::Scalar> = r_v.iter().chain(r_b.iter()).copied().collect();
    chunk_claims[bi].push(combined, reconstr_eval);

    batch_data.push(RangeCheckBatchData {
      chunk_comm: chunk_comms[bi].clone(),
      value_eval,
      reconstr_eval,
      value_reconstr_sumcheck,
    });
  }

  info!(elapsed_ms = %rcr_t.elapsed().as_millis(), "rc_reconstr");

  let chunk_data: Vec<(Vec<t256::Scalar>, HB, OpenClaims)> = chunk_fq_all
    .into_iter()
    .zip(chunk_blinds)
    .zip(chunk_claims)
    .map(|((fq, blind), claims)| (fq, blind, claims))
    .collect();

  Ok((
    SharedRangeCheck {
      mult_comm,
      logup,
      batches: batch_data,
    },
    RcProverArtifacts {
      value_claims,
      chunk_data,
      mult_fq,
      mult_blind,
      mult_claims,
    },
  ))
}

/// Verifier-side mirror of `prove_shared_range_check`. Re-derives the
/// transcript, verifies the multi-witness LogUp (with all tree depths
/// pinned to the public batch shapes), re-runs each batch's
/// reconstruction sumcheck against the claimed evaluations, and returns
/// the claims for the batched-open verification.
fn verify_shared_range_check(
  metas: &[RangeBatchMeta],
  arg: &SharedRangeCheck,
  parent: &mut Keccak256Transcript<T256DynPrimeEngine>,
) -> Result<RcVerifyClaims, SpartanError> {
  if metas.is_empty() || arg.batches.len() != metas.len() {
    return Err(SpartanError::InvalidSumcheckProof);
  }
  let dims: Vec<BatchDims> = metas
    .iter()
    .map(|m| BatchDims::new(m.num_polys, m.n_values, m.log_bound))
    .collect();

  // Tree-depth pinning: chunk trees (one per batch, sized by the public
  // batch shape), then shifted-top trees of the non-aligned batches.
  let mut expected_depths: Vec<usize> = dims.iter().map(|d| ceil_log2(d.n_chunks.max(1))).collect();
  let mut top_batches: Vec<usize> = Vec::new();
  for (bi, d) in dims.iter().enumerate() {
    if d.top_needed() {
      expected_depths.push(d.log_np + d.log_nv);
      top_batches.push(bi);
    }
  }

  // 1. Spawn the same sub-transcript the prover used.
  let mut sub = spawn_shared_range_subtranscript(
    parent,
    arg.batches.iter().map(|b| &b.chunk_comm),
    &arg.mult_comm,
  )?;

  // 2. Multi-witness LogUp membership: every chunk (and shifted top) in
  //    [0, 2^16). Its reduced claims become batched-open claims.
  let claims = arg.logup.verify(CHUNK_BITS, &expected_depths, &mut sub)?;
  let mut chunk_claims: Vec<OpenClaims> = vec![OpenClaims::default(); metas.len()];
  for (bi, (point, eval)) in claims.wit_claims.iter().take(metas.len()).enumerate() {
    chunk_claims[bi].push(point.clone(), *eval);
  }
  for (ti, &bi) in top_batches.iter().enumerate() {
    let d = &dims[bi];
    let (point, eval) = &claims.wit_claims[metas.len() + ti];
    let ext: Vec<t256::Scalar> = point
      .iter()
      .copied()
      .chain(bool_point_of_index(d.numchunks - 1, d.log_stride))
      .collect();
    chunk_claims[bi].push(ext, *eval - t256::Scalar::from(d.top_shift()));
  }
  let mut mult_claims = OpenClaims::default();
  mult_claims.push(claims.mult_point.clone(), claims.mult_eval);

  // 3. Per batch: r_v squeeze, claimed `V(r_v)` absorbed and recorded as
  //    a claim on the batch's target, reconstruction sumcheck verified
  //    against it, and the final integrand check
  //    `w(r_b)·chunk(r_v, r_b) == final claim` using the claimed chunk
  //    evaluation (itself a claim on the chunk commitment).
  let mut value_claims: Vec<(RcTarget, Vec<t256::Scalar>, t256::Scalar)> =
    Vec::with_capacity(metas.len());
  for (bi, (m, d)) in metas.iter().zip(dims.iter()).enumerate() {
    let b = &arg.batches[bi];
    let r_v: Vec<t256::Scalar> = (0..(d.log_np + d.log_nv))
      .map(|_| sub.squeeze(b"range_rv"))
      .collect::<Result<Vec<_>, _>>()?;
    sub.absorb_bytes(b"rc_value_ev", b.value_eval.to_repr().as_ref());
    value_claims.push((m.target, r_v.clone(), b.value_eval));

    let (vr_final_claim, r_b) =
      b.value_reconstr_sumcheck
        .verify(b.value_eval, d.log_stride, 2, &mut sub)?;
    sub.absorb_bytes(b"rc_reconstr_ev", b.reconstr_eval.to_repr().as_ref());

    let mut w_poly = crate::polys::multilinear::MultilinearPolynomial::new(chunk_weight_vector(
      m.log_bound,
      d.stride,
    ));
    for r in &r_b {
      w_poly.bind_poly_var_top(r);
    }
    let w_at_rb = w_poly.into_vec()[0];
    if vr_final_claim != w_at_rb * b.reconstr_eval {
      return Err(SpartanError::InvalidSumcheckProof);
    }

    let combined: Vec<t256::Scalar> = r_v.iter().chain(r_b.iter()).copied().collect();
    chunk_claims[bi].push(combined, b.reconstr_eval);
  }

  Ok(RcVerifyClaims {
    value_claims,
    chunk_claims,
    mult_claims,
  })
}

/// Absorb a `BigInt` into a `ByteTranscript` as `(sign_byte, LE
/// magnitude bytes)`. Sign byte is `0` for non-negative, `1` for
/// negative. Length-prefixed by usize → 8 bytes LE so re-derivation is
/// unambiguous.
fn absorb_bigint<T: ByteTranscript>(transcript: &mut T, x: &BigInt) {
  let sign_byte: u8 = match x.sign() {
    Sign::Minus => 1,
    _ => 0,
  };
  let mag = x.magnitude().to_bytes_le();
  let mut buf = Vec::with_capacity(1 + 8 + mag.len());
  buf.push(sign_byte);
  buf.extend_from_slice(&(mag.len() as u64).to_le_bytes());
  buf.extend_from_slice(&mag);
  transcript.absorb_bytes(b"int_v_prime", &buf);
}

#[cfg(test)]
mod tests {
  use super::*;
  use crate::dyn_prime::DynPrime;
  use crate::traits::mod_engine::SumcheckField;
  use crate::traits::transcript::TranscriptEngineTrait;

  type ME = T256DynPrimeEngine;
  type MP = IntegerModPCS;
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
    let comm = <MP as ModPCSEngineTrait<ME>>::commit(&ck, &poly, &blind).unwrap();

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

  /// `derive_optimized` returns valid params for every input length and
  /// norm bound we care about, and never ranks worse than the fixed
  /// defaults (which are inside its search space whenever they derive).
  #[test]
  fn derive_optimized_valid_and_never_worse_than_default() {
    for log_t_f in [32usize, 64, 256, 2048] {
      for num_vars in [1usize, 2, 4, 8, 12, 16, 20, 25] {
        let opt = IntEvalParams::derive_optimized(log_t_f, num_vars).unwrap();
        opt.validate(num_vars).unwrap();
        if let Ok(default) = IntEvalParams::derive_no_limb_split(log_t_f, DEFAULT_K, num_vars) {
          assert!(
            opt.estimated_prover_cost(num_vars) <= default.estimated_prover_cost(num_vars),
            "optimized params cost more than defaults for log_t_f={log_t_f}, n={num_vars}"
          );
        }
      }
    }
  }

  /// Human-readable record of what `derive_optimized` picks versus the
  /// fixed defaults, with the cost model's ratio. Documents the
  /// optimizer's behavior; the printed values are the point.
  #[test]
  fn derive_optimized_prints_table() {
    for log_t_f in [32usize, 256, 2048] {
      for num_vars in [4usize, 8, 12, 16, 20, 25] {
        let opt = IntEvalParams::derive_optimized(log_t_f, num_vars).unwrap();
        let opt_cost = opt.estimated_prover_cost(num_vars);
        let default_str = match IntEvalParams::derive_no_limb_split(log_t_f, DEFAULT_K, num_vars) {
          Ok(d) => format!(
            "default(k={}, log_P={}, s={}) cost={:.0}, ratio={:.2}",
            d.k,
            d.log_p,
            d.s,
            d.estimated_prover_cost(num_vars),
            d.estimated_prover_cost(num_vars) / opt_cost
          ),
          Err(_) => "default: invalid".to_string(),
        };
        eprintln!(
          "log_T_f={log_t_f} n={num_vars}: opt(k={}, log_P={}, s={}, log_T={}, numlimb={}) cost={opt_cost:.0} | {default_str}",
          opt.k, opt.log_p, opt.s, opt.log_t, opt.numlimb
        );
      }
    }
  }

  /// End-to-end prove/verify with `setup_optimized`-chosen params, both
  /// at a size where the optimizer can skip iterations and at one where
  /// the chain/iteration machinery runs.
  #[test]
  fn prove_verify_roundtrips_optimized_params() {
    for num_vars in [4usize, 8] {
      let n = 1usize << num_vars;
      let (ck, vk) =
        IntegerModPCS::setup_optimized(b"inteval-opt", n, 256, DEFAULT_LOG_T_F).unwrap();

      let dyn_params = small_dyn_params();
      let poly: Vec<BigUint> = (0..n).map(|i| BigUint::from(i as u32 + 1)).collect();
      let point: Vec<DP> = (0..num_vars)
        .map(|i| DP::from_u64(&dyn_params, ((i as u64) * 7 + 3) % 37))
        .collect();

      let int_point: Vec<BigUint> = point.iter().map(dyn_to_biguint).collect();
      let int_v = integer_mle_evaluate(&poly, &int_point);
      let p: BigUint = BigUint::from(37u32);
      let eval = int_v
        .mod_floor(&BigInt::from(p.clone()))
        .to_biguint()
        .unwrap();

      let blind = <MP as ModPCSEngineTrait<ME>>::blind(&ck, n);
      let comm = <MP as ModPCSEngineTrait<ME>>::commit(&ck, &poly, &blind).unwrap();

      let mut pt = <ME as SumcheckEngine>::TE::new_with_params(b"intev", dyn_params);
      let arg =
        <MP as ModPCSEngineTrait<ME>>::prove(&ck, &mut pt, &comm, &poly, &blind, &point, &eval)
          .unwrap();

      let mut vt = <ME as SumcheckEngine>::TE::new_with_params(b"intev", dyn_params);
      <MP as ModPCSEngineTrait<ME>>::verify(&vk, &mut vt, &comm, &point, &eval, &arg).unwrap();
    }
  }

  /// Wide limbs (`log_t = 128 > 64`) with large coefficients must round-trip
  /// and verify. Regression for the truncation bug: `commit` previously
  /// hardcoded `is_small = true`, so 128-bit limbs were silently truncated to
  /// their low 64 bits, producing a non-binding commitment whose opening
  /// failed the IPA verify ("first equation failed"). With the fix, wide
  /// limbs take the checked full-MSM path and the opening verifies.
  #[test]
  fn wide_limb_commit_roundtrips_and_verifies() {
    let num_vars = 6usize;
    let n = 1usize << num_vars;
    // Force log_t = 128 (numlimb = 2): the > 64-bit limb regime.
    let params = (2..16usize)
      .find_map(|k| IntEvalParams::derive(256, 128, k, num_vars).ok())
      .expect("a valid k exists for log_t = 128");
    assert!(params.log_t > 64, "test must exercise > 64-bit limbs");
    let (ck, vk) = IntegerModPCS::setup_with_params(b"wide-limb", n, 256, params).unwrap();

    let dyn_params = small_dyn_params();
    // Coefficients ~2^200 so each splits into limbs that exceed 2^64 — exactly
    // what the buggy is_small=true path truncated.
    let big = BigUint::from(2u32).pow(200);
    let poly: Vec<BigUint> = (0..n).map(|i| &big + BigUint::from(i as u32 + 1)).collect();
    let point: Vec<DP> = (0..num_vars)
      .map(|i| DP::from_u64(&dyn_params, ((i as u64) * 7 + 3) % 37))
      .collect();

    let int_point: Vec<BigUint> = point.iter().map(dyn_to_biguint).collect();
    let int_v = integer_mle_evaluate(&poly, &int_point);
    let p: BigUint = BigUint::from(37u32);
    let eval = int_v
      .mod_floor(&BigInt::from(p.clone()))
      .to_biguint()
      .unwrap();

    let blind = <MP as ModPCSEngineTrait<ME>>::blind(&ck, n);
    let comm = <MP as ModPCSEngineTrait<ME>>::commit(&ck, &poly, &blind).unwrap();

    let mut pt = <ME as SumcheckEngine>::TE::new_with_params(b"intev", dyn_params);
    let arg =
      <MP as ModPCSEngineTrait<ME>>::prove(&ck, &mut pt, &comm, &poly, &blind, &point, &eval)
        .unwrap();

    let mut vt = <ME as SumcheckEngine>::TE::new_with_params(b"intev", dyn_params);
    <MP as ModPCSEngineTrait<ME>>::verify(&vk, &mut vt, &comm, &point, &eval, &arg).unwrap();
  }

  /// `validate` catches a hand-rolled `IntEvalParams` literal where
  /// `numlimb` is inconsistent with `(log_t, log_t_f)`.
  #[test]
  fn validate_rejects_bad_numlimb() {
    let bad = IntEvalParams {
      k: 7,
      log_p: 27,
      s: 10,
      log_t: 16,
      log_t_f: 32, // ⌈32/16⌉ = 2
      numlimb: 1,  // mismatched
      numlimb_var: 0,
    };
    let err = bad.validate(8).unwrap_err();
    assert!(matches!(err, SpartanError::InvalidInputLength { .. }));
  }

  /// `numlimb` / `numlimb_var` sanity. Standard cases plus boundary.
  #[test]
  fn numlimb_basic() {
    assert_eq!(numlimb(32, 32), 1);
    assert_eq!(numlimb_var(1), 0);
    assert_eq!(numlimb(32, 16), 2);
    assert_eq!(numlimb_var(2), 1);
    assert_eq!(numlimb(32, 8), 4);
    assert_eq!(numlimb_var(4), 2);
    assert_eq!(numlimb(32, 12), 3); // ceil(32/12) = 3
    assert_eq!(numlimb_var(3), 2); // ceil(log_2 3) = 2 → pad to 4 slots
    assert_eq!(numlimb(33, 16), 3); // log_t_f not divisible by log_t
  }

  /// `chunk_decompose_value` is invertible by `sum_c 2^(16c) · chunk[c]`
  /// and every chunk lies in `[0, 2^16)`.
  #[test]
  fn chunk_decompose_round_trips() {
    for (v, log_bound) in [
      (BigUint::from(0u32), 8),
      (BigUint::from(1u32), 1),
      (BigUint::from(0xffu32), 8),
      (BigUint::from(0xabcdu32), 16),
      (BigUint::from(0xdeadbeefu32), 32),
      (BigUint::from(0xffff_ffff_ffff_ffffu64), 64),
      (BigUint::from(0x7fff_ffffu32), 31), // odd bit count, top-bit-zero
      ((BigUint::one() << 227) - BigUint::one(), 227), // b_j-style width
    ] {
      let chunks = chunk_decompose_value(&v, log_bound);
      assert_eq!(chunks.len(), log_bound.div_ceil(CHUNK_BITS));
      let rem = log_bound - CHUNK_BITS * (chunks.len() - 1);
      for (c, ch) in chunks.iter().enumerate() {
        assert!(*ch < 1u64 << CHUNK_BITS);
        if c == chunks.len() - 1 {
          assert!(*ch < 1u64 << rem, "top chunk exceeds 2^{rem}");
        }
      }
      let mut acc = BigUint::zero();
      for (c, ch) in chunks.iter().enumerate() {
        acc += BigUint::from(*ch) << (CHUNK_BITS * c);
      }
      assert_eq!(acc, v, "decomp of 0x{v:x} doesn't round-trip");
    }
  }

  /// `batch_weight`'s structured assembly (boolean-head blocks, shared-
  /// prefix tensor groups, singletons) matches the naive Σ λ^i·eq(z_i,·).
  #[test]
  fn batch_weight_matches_naive() {
    let n_vars = 6usize;
    let n = 1usize << n_vars;
    let rnd = |s: u64| t256::Scalar::from(s * s + 3);
    // Mixed structure: two boolean-head claims, a shared-prefix group of
    // three, and an unrelated singleton.
    let shared: Vec<t256::Scalar> = (0..3).map(|i| rnd(40 + i)).collect();
    let mut points: Vec<Vec<t256::Scalar>> = vec![
      bool_point_of_index(5, 3)
        .into_iter()
        .chain((0..3).map(|i| rnd(7 + i)))
        .collect(),
      bool_point_of_index(2, 2)
        .into_iter()
        .chain((0..4).map(|i| rnd(11 + i)))
        .collect(),
    ];
    for c in 0..3u64 {
      points.push(
        shared
          .iter()
          .copied()
          .chain((0..3).map(|i| rnd(100 * (c + 1) + i)))
          .collect(),
      );
    }
    points.push((0..6).map(|i| rnd(900 + i)).collect());
    let evals: Vec<t256::Scalar> = (0..points.len() as u64).map(|i| rnd(70 + i)).collect();
    let lambda = rnd(31337);

    let (w, claim) = batch_weight(&points, &evals, lambda, n);

    let mut w_naive = vec![t256::Scalar::ZERO; n];
    let mut claim_naive = t256::Scalar::ZERO;
    let mut lam = t256::Scalar::ONE;
    for (z, y) in points.iter().zip(evals.iter()) {
      let eq = EqPolynomial::<t256::Scalar>::evals_from_points(z);
      for (wj, e) in w_naive.iter_mut().zip(eq.iter()) {
        *wj += lam * e;
      }
      claim_naive += lam * y;
      lam *= lambda;
    }
    assert_eq!(w, w_naive);
    assert_eq!(claim, claim_naive);
  }

  /// `bool_point_of_index` selects the right slot: binding an MLE's
  /// variables to `bits(idx)` evaluates the dense table at `idx`.
  #[test]
  fn bool_point_selects_index() {
    let table: Vec<t256::Scalar> = (0..8u64).map(|i| t256::Scalar::from(100 + i)).collect();
    for idx in 0..8usize {
      let pt = bool_point_of_index(idx, 3);
      assert_eq!(mle_evaluate_fq(&table, &pt), table[idx]);
    }
  }

  /// `split_value_into_limbs` is invertible: reconstruct from limbs.
  #[test]
  fn split_value_round_trips() {
    let log_t = 8usize;
    let t = BigUint::one() << log_t;
    for (v, log_t_f) in [
      (BigUint::from(0u32), 32),
      (BigUint::from(1u32), 32),
      (BigUint::from(0xdeadbeefu32), 32),
      (BigUint::from(0xffffu32), 16),
      (BigUint::from(0xffu32), 8),
      (BigUint::from(0xffff_ffff_ffff_ffffu64), 64),
    ] {
      let nl = numlimb(log_t_f, log_t);
      let limbs = split_value_into_limbs(&v, log_t, nl);
      assert_eq!(limbs.len(), nl);
      for limb in &limbs {
        assert!(limb < &t, "limb 0x{:x} exceeds 2^{}", limb, log_t);
      }
      // Reconstruct: sum_i T^i · limbs[i].
      let mut acc = BigUint::zero();
      for limb in limbs.iter().rev() {
        acc = &acc * &t + limb;
      }
      assert_eq!(acc, v);
    }
  }

  /// `limb_split_polynomial` no-op when `log_t == log_t_f` (numlimb=1).
  /// In that case the output equals the input (only one limb, no
  /// padding since `numlimb_var = 0` → stride = 1).
  #[test]
  fn limb_split_no_op_when_log_t_eq_log_t_f() {
    let poly: Vec<BigUint> = (0..8u32).map(BigUint::from).collect();
    let out = limb_split_polynomial(&poly, 32, 32);
    assert_eq!(out, poly);
  }

  /// `limb_split_polynomial` with `numlimb = 2` (T = 2^8, T_f = 2^16):
  /// each coefficient becomes two limbs `(low, high)`, laid out in
  /// adjacent slots. Recoverable by `low + 256 · high == original`.
  #[test]
  fn limb_split_pairs_of_limbs() {
    let poly = vec![
      BigUint::from(0x0000u32),
      BigUint::from(0x00ffu32),
      BigUint::from(0xff00u32),
      BigUint::from(0xabcdu32),
    ];
    let out = limb_split_polynomial(&poly, 8, 16);
    assert_eq!(out.len(), 8); // 4 · 2 slots

    for (x, orig) in poly.iter().enumerate() {
      let lo = &out[x * 2];
      let hi = &out[x * 2 + 1];
      let reconstructed = lo + BigUint::from(256u32) * hi;
      assert_eq!(&reconstructed, orig, "slot {x}");
    }
  }

  /// `limb_split_polynomial` with non-power-of-two `numlimb`: pad
  /// the missing slots with zero.
  #[test]
  fn limb_split_pads_to_power_of_two() {
    let poly = vec![BigUint::from(0x0afbcu32)]; // 20 bits
    let out = limb_split_polynomial(&poly, 8, 20); // numlimb = 3, stride = 4
    assert_eq!(out.len(), 4);
    // 0x0afbc = 0xbc + 0xaf · 256 + 0x00 · 65536.
    assert_eq!(out[0], BigUint::from(0xbcu32));
    assert_eq!(out[1], BigUint::from(0xafu32));
    assert_eq!(out[2], BigUint::from(0x00u32)); // top limb (within numlimb)
    assert_eq!(out[3], BigUint::from(0u32)); // padding slot
  }

  /// Truncated divmod gives symmetric `(q, r)`: `divmod(-g) = (-q, -r)`.
  #[test]
  fn truncated_divmod_is_symmetric() {
    for (g, d, eq, er) in [
      (7i64, 2u64, 3i64, 1i64),
      (-7, 2, -3, -1),
      (8, 2, 4, 0),
      (-8, 2, -4, 0),
      (-7, 3, -2, -1),
      (0, 5, 0, 0),
    ] {
      let (q, r) = truncated_divmod(&BigInt::from(g), &BigUint::from(d));
      assert_eq!(q, BigInt::from(eq), "q wrong for {g} / {d}");
      assert_eq!(r, BigInt::from(er), "r wrong for {g} mod {d}");
      // Identity always holds.
      assert_eq!(&q * BigInt::from(d) + &r, BigInt::from(g));
    }
  }

  /// Partial-eval at the last variable should match a 2-step direct
  /// evaluation: poly is 8 evals (3 vars), partial-eval the last var,
  /// then evaluate the remaining 2-var poly at a 2-component point.
  #[test]
  fn integer_partial_evaluate_matches_full_eval() {
    // poly[x_0, x_1, x_2] = 100·x_0 + 10·x_1 + x_2 (over Z).
    // The evaluation table walks (x_0, x_1, x_2) in big-endian bit order,
    // so poly[(b2 b1 b0)] = 100·b2 + 10·b1 + b0.
    let poly: Vec<BigInt> = (0..8u32)
      .map(|k| BigInt::from(100 * ((k >> 2) & 1) + 10 * ((k >> 1) & 1) + (k & 1)))
      .collect();
    // Partial-eval at last variable to value 3.
    let r_last = vec![BigUint::from(3u32)];
    let g = integer_partial_evaluate_top_k(&poly, &r_last);
    assert_eq!(g.len(), 4);
    // g[(b2 b1)] = poly(b2, b1, 3) = 100·b2 + 10·b1 + 3.
    for k in 0..4u32 {
      let expected = BigInt::from(100 * ((k >> 1) & 1) + 10 * (k & 1) + 3);
      assert_eq!(g[k as usize], expected);
    }
  }

  /// `integer_mle_evaluate`'s fold matches the naive chi-product sum
  /// (the pre-optimization definition), pinning the variable bit-order.
  #[test]
  fn integer_mle_evaluate_matches_naive() {
    let num_vars = 5usize;
    let n = 1usize << num_vars;
    let poly: Vec<BigUint> = (0..n).map(|k| BigUint::from((k * k + 7) as u64)).collect();
    let point: Vec<BigUint> = (0..num_vars)
      .map(|i| BigUint::from((1u64 << 40) + 31 * i as u64 + 5))
      .collect();

    // Naive: sum_k chi(point, k) · poly[k], variable i ↔ bit num_vars−1−i.
    let point_int: Vec<BigInt> = point.iter().map(|x| BigInt::from(x.clone())).collect();
    let one = BigInt::one();
    let mut naive = BigInt::zero();
    for (k, poly_k) in poly.iter().enumerate() {
      let mut chi = one.clone();
      for (i, pi) in point_int.iter().enumerate() {
        let bit = (k >> (num_vars - 1 - i)) & 1;
        chi *= if bit == 1 { pi.clone() } else { &one - pi };
      }
      naive += chi * BigInt::from(poly_k.clone());
    }

    assert_eq!(integer_mle_evaluate(&poly, &point), naive);
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
    let (_ck, _vk) = IntegerModPCS::setup_with_params(b"override", n, 256, p).unwrap();

    // Bad params: zero `s` makes soundness_1 fail trivially.
    let bad = IntEvalParams {
      k: 7,
      log_p: 20,
      s: 0,
      log_t: 32,
      log_t_f: 32,
      numlimb: 1,
      numlimb_var: 0,
    };
    let err = IntegerModPCS::setup_with_params(b"override", n, 256, bad).unwrap_err();
    assert!(matches!(err, SpartanError::InvalidInputLength { .. }));
  }

  /// Helper: build params for a small dynamic prime so we can
  /// deterministically evaluate the polynomial at a known Z_p point.
  fn small_dyn_params() -> crypto_bigint::modular::FixedMontyParams<4> {
    use crypto_bigint::{Odd, U256};
    // A small prime (37) so the integer evaluation is human-verifiable.
    crypto_bigint::modular::FixedMontyParams::new(Odd::new(U256::from(37u32)).unwrap())
  }

  /// End-to-end IntEval prove/verify for the `n ≤ k` regime.
  #[test]
  fn prove_verify_roundtrips_small_witness() {
    let num_vars = 4usize;
    let n = 1usize << num_vars; // 16
    let (ck, vk) = <MP as ModPCSEngineTrait<ME>>::setup(b"inteval-rt", n, 256);

    let dyn_params = small_dyn_params();
    let poly: Vec<BigUint> = (0..n).map(|i| BigUint::from(i as u32 + 1)).collect();
    let point: Vec<DP> = (0..num_vars)
      .map(|i| DP::from_u64(&dyn_params, ((i as u64) * 7 + 3) % 37))
      .collect();

    // Oracle Z_p eval: take the integer evaluation reduced mod p.
    let int_point: Vec<BigUint> = point.iter().map(dyn_to_biguint).collect();
    let int_v = integer_mle_evaluate(&poly, &int_point);
    let p: BigUint = BigUint::from(37u32);
    let eval = int_v
      .mod_floor(&BigInt::from(p.clone()))
      .to_biguint()
      .unwrap();

    let blind = <MP as ModPCSEngineTrait<ME>>::blind(&ck, n);
    let comm = <MP as ModPCSEngineTrait<ME>>::commit(&ck, &poly, &blind).unwrap();

    let mut pt = <ME as SumcheckEngine>::TE::new_with_params(b"intev", dyn_params);
    let arg =
      <MP as ModPCSEngineTrait<ME>>::prove(&ck, &mut pt, &comm, &poly, &blind, &point, &eval)
        .unwrap();

    let mut vt = <ME as SumcheckEngine>::TE::new_with_params(b"intev", dyn_params);
    <MP as ModPCSEngineTrait<ME>>::verify(&vk, &mut vt, &comm, &point, &eval, &arg).unwrap();
  }

  /// Verifier rejects a tampered claimed Z_p eval.
  #[test]
  fn verify_rejects_wrong_eval() {
    let num_vars = 4usize;
    let n = 1usize << num_vars;
    let (ck, vk) = <MP as ModPCSEngineTrait<ME>>::setup(b"inteval-rt", n, 256);

    let dyn_params = small_dyn_params();
    let poly: Vec<BigUint> = (0..n).map(|i| BigUint::from(i as u32 + 1)).collect();
    let point: Vec<DP> = (0..num_vars)
      .map(|i| DP::from_u64(&dyn_params, ((i as u64) * 7 + 3) % 37))
      .collect();

    let int_point: Vec<BigUint> = point.iter().map(dyn_to_biguint).collect();
    let int_v = integer_mle_evaluate(&poly, &int_point);
    let p = BigUint::from(37u32);
    let real_eval = int_v
      .mod_floor(&BigInt::from(p.clone()))
      .to_biguint()
      .unwrap();
    // Tamper: add 1 mod 37.
    let bad_eval = (real_eval.clone() + BigUint::from(1u32)) % &p;

    let blind = <MP as ModPCSEngineTrait<ME>>::blind(&ck, n);
    let comm = <MP as ModPCSEngineTrait<ME>>::commit(&ck, &poly, &blind).unwrap();

    let mut pt = <ME as SumcheckEngine>::TE::new_with_params(b"intev", dyn_params);
    let arg =
      <MP as ModPCSEngineTrait<ME>>::prove(&ck, &mut pt, &comm, &poly, &blind, &point, &real_eval)
        .unwrap();

    // Verifier with the bad eval claim must reject.
    let mut vt = <ME as SumcheckEngine>::TE::new_with_params(b"intev", dyn_params);
    let err = <MP as ModPCSEngineTrait<ME>>::verify(&vk, &mut vt, &comm, &point, &bad_eval, &arg)
      .unwrap_err();
    assert!(matches!(err, SpartanError::InvalidSumcheckProof));
  }

  /// Step C end-to-end: `n > k` triggers the partial-eval iteration
  /// path. Uses explicit small-k params (k=2) so a 4-var poly hits
  /// `t = ⌈(4-2)/2⌉ = 1` iteration.
  #[test]
  fn prove_verify_roundtrips_with_iteration() {
    let num_vars = 4usize;
    let n = 1usize << num_vars; // 16
    // Small-k config so the partial-eval iteration path triggers.
    // `derive` picks the largest valid log_p and the smallest s.
    let small_params =
      IntEvalParams::derive_no_limb_split(8, 2, num_vars).expect("valid derived params");
    let (ck, vk) = IntegerModPCS::setup_with_params(b"inteval-iter", n, 256, small_params).unwrap();

    let dyn_params = small_dyn_params();
    let poly: Vec<BigUint> = (0..n).map(|i| BigUint::from(i as u32 + 1)).collect();
    let point: Vec<DP> = (0..num_vars)
      .map(|i| DP::from_u64(&dyn_params, ((i as u64) * 3 + 5) % 37))
      .collect();

    let int_point: Vec<BigUint> = point.iter().map(dyn_to_biguint).collect();
    let int_v = integer_mle_evaluate(&poly, &int_point);
    let p = BigUint::from(37u32);
    let eval = int_v
      .mod_floor(&BigInt::from(p.clone()))
      .to_biguint()
      .unwrap();

    let blind = <MP as ModPCSEngineTrait<ME>>::blind(&ck, n);
    let comm = <MP as ModPCSEngineTrait<ME>>::commit(&ck, &poly, &blind).unwrap();

    let mut pt = <ME as SumcheckEngine>::TE::new_with_params(b"intev-iter", dyn_params);
    let arg =
      <MP as ModPCSEngineTrait<ME>>::prove(&ck, &mut pt, &comm, &poly, &blind, &point, &eval)
        .unwrap();

    let mut vt = <ME as SumcheckEngine>::TE::new_with_params(b"intev-iter", dyn_params);
    <MP as ModPCSEngineTrait<ME>>::verify(&vk, &mut vt, &comm, &point, &eval, &arg).unwrap();
  }

  /// Two-iteration roundtrip (`k=2`, `num_vars=6` → `t=2`). Exercises the
  /// a_prev batch (j=1, batched across chains) *and* the per-iteration
  /// individual a_prev opens (j=2, chain-specific commitments) together.
  #[test]
  fn prove_verify_roundtrips_with_two_iterations() {
    let num_vars = 6usize;
    let n = 1usize << num_vars; // 64
    let small_params =
      IntEvalParams::derive_no_limb_split(8, 2, num_vars).expect("valid derived params");
    let (ck, vk) =
      IntegerModPCS::setup_with_params(b"inteval-iter2", n, 256, small_params).unwrap();

    let dyn_params = small_dyn_params();
    let poly: Vec<BigUint> = (0..n).map(|i| BigUint::from(i as u32 + 1)).collect();
    let point: Vec<DP> = (0..num_vars)
      .map(|i| DP::from_u64(&dyn_params, ((i as u64) * 3 + 5) % 37))
      .collect();
    let int_point: Vec<BigUint> = point.iter().map(dyn_to_biguint).collect();
    let eval = integer_mle_evaluate(&poly, &int_point)
      .mod_floor(&BigInt::from(37u32))
      .to_biguint()
      .unwrap();

    let blind = <MP as ModPCSEngineTrait<ME>>::blind(&ck, n);
    let comm = <MP as ModPCSEngineTrait<ME>>::commit(&ck, &poly, &blind).unwrap();

    let mut pt = <ME as SumcheckEngine>::TE::new_with_params(b"intev-iter2", dyn_params);
    let arg =
      <MP as ModPCSEngineTrait<ME>>::prove(&ck, &mut pt, &comm, &poly, &blind, &point, &eval)
        .unwrap();

    // Confirm we actually exercised t=2 (two stacked layers, two layer
    // batched opens: f + ab_1 + ab_2 + chunks + mult).
    assert_eq!(arg.chains[0].iterations.len(), 2, "expected t=2");
    assert_eq!(arg.ab_comms.len(), 2, "expected 2 stacked layer comms");

    let mut vt = <ME as SumcheckEngine>::TE::new_with_params(b"intev-iter2", dyn_params);
    <MP as ModPCSEngineTrait<ME>>::verify(&vk, &mut vt, &comm, &point, &eval, &arg).unwrap();
  }

  /// Step D5 (stacked rbatchrange): tampering *any* range-check group's
  /// committed bit evaluation must make the verifier reject. The
  /// iteration config (k=2, num_vars=4 → t=1) yields three groups:
  /// `f_limb`, the `a_1` batch, and the `b_1` batch — so this exercises
  /// every segment type. A passing roundtrip with all groups present is
  /// not enough; we must confirm each group is actually checked.
  #[test]
  fn verify_rejects_tampered_range_check() {
    let num_vars = 4usize;
    let n = 1usize << num_vars;
    let small_params =
      IntEvalParams::derive_no_limb_split(8, 2, num_vars).expect("valid derived params");
    let (ck, vk) =
      IntegerModPCS::setup_with_params(b"inteval-rc-tamper", n, 256, small_params).unwrap();

    let dyn_params = small_dyn_params();
    let poly: Vec<BigUint> = (0..n).map(|i| BigUint::from(i as u32 + 1)).collect();
    let point: Vec<DP> = (0..num_vars)
      .map(|i| DP::from_u64(&dyn_params, ((i as u64) * 3 + 5) % 37))
      .collect();
    let int_point: Vec<BigUint> = point.iter().map(dyn_to_biguint).collect();
    let eval = integer_mle_evaluate(&poly, &int_point)
      .mod_floor(&BigInt::from(37u32))
      .to_biguint()
      .unwrap();

    let blind = <MP as ModPCSEngineTrait<ME>>::blind(&ck, n);
    let comm = <MP as ModPCSEngineTrait<ME>>::commit(&ck, &poly, &blind).unwrap();

    let mut pt = <ME as SumcheckEngine>::TE::new_with_params(b"intev-rc", dyn_params);
    let arg =
      <MP as ModPCSEngineTrait<ME>>::prove(&ck, &mut pt, &comm, &poly, &blind, &point, &eval)
        .unwrap();

    // Three batches (f_limb, a_1, b_1) — the config must produce them.
    assert_eq!(
      arg.range_check.batches.len(),
      3,
      "expected f_limb + a_1 + b_1 batches"
    );

    // Tampering each batch's chunk opening (in turn) must be rejected
    // (the Hyrax opening check or the LogUp wit-eval check fires,
    // depending on which trips first).
    for gi in 0..arg.range_check.batches.len() {
      let mut bad = arg.clone();
      bad.range_check.batches[gi].value_eval += t256::Scalar::ONE;
      let mut vt = <ME as SumcheckEngine>::TE::new_with_params(b"intev-rc", dyn_params);
      assert!(
        <MP as ModPCSEngineTrait<ME>>::verify(&vk, &mut vt, &comm, &point, &eval, &bad,).is_err(),
        "batch {gi} tamper not rejected"
      );
    }

    // Tampering a batched open's final evaluation must be rejected.
    let mut bad = arg.clone();
    bad.combined_open.final_evals[0] += t256::Scalar::ONE;
    let mut vt = <ME as SumcheckEngine>::TE::new_with_params(b"intev-rc", dyn_params);
    assert!(
      <MP as ModPCSEngineTrait<ME>>::verify(&vk, &mut vt, &comm, &point, &eval, &bad,).is_err()
    );

    // Dropping a batch (count mismatch) must also be rejected.
    let mut short = arg.clone();
    short.range_check.batches.pop();
    let mut vt = <ME as SumcheckEngine>::TE::new_with_params(b"intev-rc", dyn_params);
    assert!(
      <MP as ModPCSEngineTrait<ME>>::verify(&vk, &mut vt, &comm, &point, &eval, &short,).is_err()
    );
  }

  /// Step D4 end-to-end: real limb-splitting (`log_T < log_T_f` → `numlimb
  /// = 2`, `numlimb_var = 1`). Each polynomial coefficient is split into
  /// two 4-bit limbs; the F-PCS commits a polynomial of `2 · n = 32`
  /// slots. The reduction sumcheck runs one round and binds `r_k` of
  /// length 1; the IntEval body operates on `f_limb` at the extended
  /// point `(int_r, int_r_k)`.
  #[test]
  fn prove_verify_roundtrips_with_limb_split() {
    let num_vars = 4usize;
    let n = 1usize << num_vars;

    // log_T = 4 < log_T_f = 8 → numlimb = 2, numlimb_var = 1.
    // k = 2 keeps soundness derivation feasible at this small λ-style
    // setup. Coefficients < 2^8 fit in two 4-bit limbs.
    let limb_params = IntEvalParams::derive(8, 4, 2, num_vars).expect("valid derived params");
    assert_eq!(limb_params.numlimb, 2);
    assert_eq!(limb_params.numlimb_var, 1);

    let (ck, vk) =
      IntegerModPCS::setup_with_params(b"limb-split-test", n, 256, limb_params).unwrap();

    let dyn_params = small_dyn_params();
    // Coefficients in [0, 2^8). The integer eval can grow large but
    // mod p reduces to a clean Z_p value.
    let poly: Vec<BigUint> = (0..n)
      .map(|i| BigUint::from((i * 13 + 1) as u32 & 0xff))
      .collect();
    let point: Vec<DP> = (0..num_vars)
      .map(|i| DP::from_u64(&dyn_params, ((i as u64) * 7 + 2) % 37))
      .collect();

    let int_point: Vec<BigUint> = point.iter().map(dyn_to_biguint).collect();
    let int_v = integer_mle_evaluate(&poly, &int_point);
    let p = BigUint::from(37u32);
    let eval = int_v
      .mod_floor(&BigInt::from(p.clone()))
      .to_biguint()
      .unwrap();

    let blind = <MP as ModPCSEngineTrait<ME>>::blind(&ck, n);
    let comm = <MP as ModPCSEngineTrait<ME>>::commit(&ck, &poly, &blind).unwrap();

    let mut pt = <ME as SumcheckEngine>::TE::new_with_params(b"limb-split", dyn_params);
    let arg =
      <MP as ModPCSEngineTrait<ME>>::prove(&ck, &mut pt, &comm, &poly, &blind, &point, &eval)
        .unwrap();
    // The reduction sumcheck ran one round → one entry in
    // reduction_round_polys.
    assert_eq!(arg.reduction_round_polys.len(), 1);

    let mut vt = <ME as SumcheckEngine>::TE::new_with_params(b"limb-split", dyn_params);
    <MP as ModPCSEngineTrait<ME>>::verify(&vk, &mut vt, &comm, &point, &eval, &arg).unwrap();
  }

  /// Regression: limb-split commit when the *inflated* polynomial spans
  /// multiple Hyrax rows (`width < 2^numlimb_var · n`). `blind` must cover
  /// the inflated length, not the input length — otherwise `commit`
  /// indexes past the blind. The masked case from
  /// `prove_verify_roundtrips_with_limb_split` (where `n < width`, so
  /// everything fit in one row) did not exercise this.
  #[test]
  fn limb_split_commit_spans_multiple_hyrax_rows() {
    let num_vars = 4usize;
    let n = 1usize << num_vars; // 16
    // numlimb = 2, numlimb_var = 1 → inflated length 32.
    let limb_params = IntEvalParams::derive(8, 4, 2, num_vars).expect("valid derived params");
    assert_eq!(limb_params.numlimb_var, 1);

    // width = 4 < inflated length 32 → div_ceil(32, 4) = 8 Hyrax rows,
    // versus div_ceil(16, 4) = 4 rows for the un-inflated blind.
    let (ck, _vk) =
      IntegerModPCS::setup_with_params(b"limb-split-rows", n, 4, limb_params).unwrap();

    let poly: Vec<BigUint> = (0..n)
      .map(|i| BigUint::from((i * 13 + 1) as u32 & 0xff))
      .collect();

    let blind = <MP as ModPCSEngineTrait<ME>>::blind(&ck, n);
    // The bug manifested as an index-out-of-bounds panic here.
    let comm = <MP as ModPCSEngineTrait<ME>>::commit(&ck, &poly, &blind).unwrap();

    // Commit matches a direct Hyrax commit of the limb-split polynomial.
    let limbs = limb_split_polynomial(&poly, 4, 8);
    let limbs_fq: Vec<t256::Scalar> = limbs.iter().map(biguint_to_scalar).collect();
    let direct = Hyrax::commit(&ck.inner, &limbs_fq, &blind.inner, false).unwrap();
    assert_eq!(comm.inner, direct);
  }
}
