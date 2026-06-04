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
    T256DynPrimeEngine, T256HyraxEngine, keccak::Keccak256Transcript, pcs::hyrax_pc::HyraxPCS,
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
    let log_n = ceil_log2(num_vars_total.max(1));
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
    let log_n = ceil_log2(num_vars_total.max(1));
    let log_lambda = ceil_log2(LAMBDA);

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
pub struct IntegerModCommitmentKey {
  pub(crate) inner: <Hyrax as PCSEngineTrait<T256HyraxEngine>>::CommitmentKey,
  pub(crate) params: IntEvalParams,
}

/// Verifier key wraps Hyrax's plus the IntEval parameters.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct IntegerModVerifierKey {
  pub(crate) inner: <Hyrax as PCSEngineTrait<T256HyraxEngine>>::VerifierKey,
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

/// One per-small-prime opening: the F-side evaluation `F_y^(i)`, the
/// blind used to commit it, and the Hyrax evaluation argument for the
/// opening at the small-prime-reduced point.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct SmallPrimeOpening {
  /// F_y^(i) = f_F(r mod p_i), the F-evaluation at the small-prime-
  /// reduced point. Sent in the clear; verifier checks it for the
  /// CRT congruence `to_int(F_y^(i)) ≡ int_v' (mod p_i)`.
  pub f_y: t256::Scalar,
  /// Blind used to commit `f_y`. Verifier reconstructs `comm_eval_i`
  /// from `(f_y, blind_eval)` and feeds it to `Hyrax::verify`.
  pub blind_eval: <Hyrax as PCSEngineTrait<T256HyraxEngine>>::Blind,
  /// Hyrax evaluation argument for the opening at `r mod p_i`.
  pub hyrax_arg: <Hyrax as PCSEngineTrait<T256HyraxEngine>>::EvaluationArgument,
}

/// One iteration's identity-check eval claims at γ. The quotient `a_j` and
/// remainder `b_j` are no longer committed individually — they live as
/// zero-padded blocks of the stacked commitments `F_a` / `F_b` (see
/// [`IntEvalArgument::comm_f_a`]). Only the eval scalars consumed by the
/// per-iteration identity check are stored here; their binding to
/// `F_a`/`F_b` is via the multi-point batches
/// [`IntEvalArgument::f_a_batch`] / [`IntEvalArgument::f_b_batch`].
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct IterationOracles {
  /// Claimed `a_{j-1}(γ_ext)` where `γ_ext = (γ[0..n-jk], r^(i)[n-jk..n-(j-1)k])`.
  /// Used directly in the identity check. For `j>1` it's an evaluation of
  /// the `a_{j-1}` block of `F_a` (bound by the F_a batch); for `j=1` it's
  /// an evaluation of the input commitment (bound by `a_prev_batch`).
  pub a_prev_eval: t256::Scalar,
  /// Claimed `a_j(γ[0..n-jk])`. Used in the identity check; bound to `F_a`
  /// by the F_a multi-point batch.
  pub a_curr_eval: t256::Scalar,
  /// Claimed `b_j(γ[0..n-jk])`. Used in the identity check; bound to `F_b`
  /// by the F_b multi-point batch.
  pub b_curr_eval: t256::Scalar,
}

/// Per-prime chain: `t = ⌈(n-k)/k⌉` iterations plus the claimed final-
/// remainder evaluation `a_t(r^(i)[0..n-tk])`. The commitments to
/// `a_j`/`b_j` live in the shared `F_a`/`F_b` stacks; the per-iteration
/// and final eval claims are bound by the F_a/F_b multi-point batches. For
/// `n ≤ k` the iterations vec is empty and `final_eval = f(r^(i))` (bound
/// by the `t=0` input batch in `a_prev_batch`).
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ChainData {
  /// Per-iteration identity-check eval claims.
  pub iterations: Vec<IterationOracles>,
  /// Claimed `a_t(r^(i)[0..n-tk])` (or `f(r^(i))` when `t=0`). Sent in the
  /// clear, checked by the CRT congruence; bound by the F_a batch (`t>0`)
  /// or the input batch (`t=0`).
  pub final_eval: t256::Scalar,
}

/// Evaluation argument: the prover-sent integer evaluation `int_v'`,
/// the reduction-sumcheck round polynomials (Phase-3 step D3), the stacked
/// `F_a`/`F_b` commitments, range checks, and the multi-point batches.
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
  /// Stacked commitment to all (shifted) `a_j^c` blocks, each zero-padded
  /// to size `2^{n-k}` and placed at block `p(c,j)=c*t+(j-1)`. Committed
  /// once and absorbed before γ. `None` when `t=0`.
  pub(crate) comm_f_a: Option<<Hyrax as PCSEngineTrait<T256HyraxEngine>>::Commitment>,
  /// Stacked commitment to all (shifted) `b_j^c` blocks. `None` when `t=0`.
  pub(crate) comm_f_b: Option<<Hyrax as PCSEngineTrait<T256HyraxEngine>>::Commitment>,
  /// Phase-3 step D5 range checks: `[f_limb, F_a, F_b]` for `t>0`, just
  /// `[f_limb]` for `t=0`. The `F_a`/`F_b` groups range-check every block
  /// of the corresponding stack in one argument (all `a_j` share bound
  /// `2^{log_p+1}`, all `b_j` share `2^{LOG_Q-log_p+1}`). See
  /// [`prove_batch_range_check`].
  pub(crate) range_checks: Vec<BatchRangeCheck>,
  /// Multi-point batch binding all `a_j^c` opens to `F_a`: per chain, each
  /// `a_j` curr-open at `γ[0..n-jk]`, each `a_{j-1}` a_prev-open at γ_ext
  /// (`j>1`), and the final `a_t` open at `r^(i)`. `None` when `t=0`.
  pub(crate) f_a_batch: Option<MultiPointBatch>,
  /// Multi-point batch binding all `b_j^c` curr opens (`b_j` at
  /// `γ[0..n-jk]`) to `F_b`. `None` when `t=0`.
  pub(crate) f_b_batch: Option<MultiPointBatch>,
  /// Multi-point batch against the input commitment `f`. For `t>0` it
  /// batches the `s` chains' `j=1` `a_prev` opens (input `f` at distinct
  /// points). For `t=0` it instead batches the `s` final opens
  /// `f(r^(i)_c)`. `None` only when there are no chains.
  pub(crate) a_prev_batch: Option<MultiPointBatch>,
}

/// Multi-point batch evaluation of one committed polynomial `F` at many
/// points `{POINT_i}` with claimed evals `{y_i}`: proves
/// `Σ_i λ^i·F(POINT_i) = Σ_x F(x)·W(x)` with `W = Σ_i λ^i·eq(POINT_i,·)`
/// via one degree-2 sumcheck reducing to a single opening of `F` at the
/// sumcheck challenge `r`. Used for the `F_a`/`F_b` stacks and the
/// input-commitment `a_prev`/final batch.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct MultiPointBatch {
  /// Degree-2 sumcheck on `F(x)·W(x)`, over the Hyrax base field.
  pub(crate) sumcheck: crate::sumcheck::SumcheckProof<T256HyraxEngine>,
  /// Opening of `F` at the sumcheck challenge `r`.
  pub(crate) f_open: SmallPrimeOpening,
}

/// Batched range-check argument (paper's `rbatchrange`) for a
/// *homogeneous batch* of `N` value polynomials — all of the same
/// length `n_values` and the same bound `2^log_bound`. The `N` polys
/// are stacked along a top "poly-index" axis into one bit polynomial
/// of `N_pad · n_values · 2^log_log_bound` bits (`N_pad = next_pow2(N)`),
/// committed once. The protocol runs one bit-validity zerocheck and one
/// value-reconstruction sumcheck over the whole stack, plus two openings
/// of the bit polynomial and one opening per value polynomial.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct BatchRangeCheck {
  /// Stacked bit-polynomial commitment.
  pub(crate) bit_comm: <Hyrax as PCSEngineTrait<T256HyraxEngine>>::Commitment,
  /// Bit-validity sumcheck (`bit · (1 - bit) = 0`), over the Hyrax
  /// base field.
  pub(crate) bit_validity_sumcheck: crate::sumcheck::SumcheckProof<T256HyraxEngine>,
  /// Value-reconstruction sumcheck (`sum_b 2^b · bit(r_v, b) = value(r_v)`),
  /// over the Hyrax base field.
  pub(crate) value_reconstr_sumcheck: crate::sumcheck::SumcheckProof<T256HyraxEngine>,
  /// Single opening of the `eq(r_v_poly, ·)`-folded value commitment at
  /// the within-poly part `r_v_within`. Its evaluation is
  /// `V(r_v) = Σ_p eq(r_v_poly, p)·value_p(r_v_within)` — one Hyrax open
  /// for the whole batch (the per-poly commitments are folded
  /// homomorphically via `fold_commitments`).
  pub(crate) value_open_at_rv: SmallPrimeOpening,
  /// Opening of the bit polynomial at the bit-validity sumcheck's
  /// final challenge point.
  pub(crate) bit_open_validity_final: SmallPrimeOpening,
  /// Opening of the bit polynomial at `(r_v, r_b)` — the value-
  /// reconstruction sumcheck's final point combining `r_v` (poly-index
  /// ++ within) and `r_b` (the b-axis sumcheck challenges).
  pub(crate) bit_open_reconstr_final: SmallPrimeOpening,
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

/// Decompose a `BigUint` value `v ∈ [0, 2^num_bits)` into its
/// little-endian bit representation. Output `bits[i] ∈ {0, 1}` with
/// `v = sum_i 2^i · bits[i]`. Asserts `v < 2^num_bits`; values that
/// exceed the bound are caller errors. Used by the Phase-3 step D5
/// batch range-check arguments to prove `value < 2^num_bits` via a
/// sumcheck on bit constraints + value reconstruction.
fn bit_decompose_value(v: &BigUint, num_bits: usize) -> Vec<u8> {
  let mut out = Vec::with_capacity(num_bits);
  let bytes = v.to_bytes_le();
  for i in 0..num_bits {
    let byte_idx = i / 8;
    let bit_in_byte = i % 8;
    let bit = if byte_idx < bytes.len() {
      (bytes[byte_idx] >> bit_in_byte) & 1
    } else {
      0
    };
    out.push(bit);
  }
  debug_assert!(
    bit_decompose_check_no_overflow(&bytes, num_bits),
    "value 0x{:x} exceeds bound 2^{}",
    v,
    num_bits
  );
  out
}

/// Helper for `bit_decompose_value`'s debug_assert: checks that the
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

/// Flatten a batch of values into a single bit polynomial of length
/// `values.len() * num_bits`, with each value's bits stored
/// contiguously in little-endian order: `bits[i * num_bits + b]` is the
/// `b`-th bit of `values[i]`. Used as the witness polynomial of the
/// batched range-check sumcheck (step D5).
fn bit_decompose_polynomial(values: &[BigUint], num_bits: usize) -> Vec<u8> {
  values
    .par_iter()
    .flat_map_iter(|v| bit_decompose_value(v, num_bits).into_iter())
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

  // Pre-lift point components to BigInt.
  let point_int: Vec<BigInt> = point.iter().map(|x| BigInt::from(x.clone())).collect();
  let one = BigInt::one();

  // Walk all 2^num_vars hypercube points. Bit-order matches
  // `EqPolynomial::evals_from_points`: variable `i ∈ [0, num_vars)`
  // corresponds to bit `num_vars - 1 - i` of `k`.
  let mut acc = BigInt::zero();
  for (k, poly_k) in poly.iter().enumerate().take(n) {
    let mut chi = one.clone();
    for (i, pi) in point_int.iter().enumerate().take(num_vars) {
      let bit = (k >> (num_vars - 1 - i)) & 1;
      let factor = if bit == 1 { pi.clone() } else { &one - pi };
      chi *= factor;
    }
    acc += chi * BigInt::from(poly_k.clone());
  }
  acc
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
    Ok((
      IntegerModCommitmentKey {
        inner: inner_ck,
        params: params.clone(),
      },
      IntegerModVerifierKey {
        inner: inner_vk,
        params,
      },
    ))
  }
}

impl ModPCSEngineTrait<T256DynPrimeEngine> for IntegerModPCS {
  type CommitmentKey = IntegerModCommitmentKey;
  type VerifierKey = IntegerModVerifierKey;
  type Commitment = IntegerModCommitment;
  type Blind = IntegerModBlind;
  type EvaluationArgument = IntEvalArgument;

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
    let inflated_n = n
      .checked_shl(params.numlimb_var as u32)
      .expect("n * 2^numlimb_var overflows usize");
    let (inner_ck, inner_vk) = Hyrax::setup(label, inflated_n, width);
    (
      IntegerModCommitmentKey {
        inner: inner_ck,
        params: params.clone(),
      },
      IntegerModVerifierKey {
        inner: inner_vk,
        params,
      },
    )
  }

  fn precompute_ck(ck: &Self::CommitmentKey) {
    Hyrax::precompute_ck(&ck.inner)
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
    is_small: bool,
  ) -> Result<Self::Commitment, SpartanError> {
    // Limb-split the integer-valued polynomial: each coefficient
    // becomes `numlimb` limbs each in `[0, T)`. For `numlimb = 1`
    // (no-limb-split mode) this returns `v` unchanged. The underlying
    // F PCS commits the limb polynomial, which has `numlimb * v.len()`
    // (or `2^numlimb_var * v.len()` after padding) coefficients.
    //
    // Stopgap: size-1 commits are *eval-value* commits issued by the
    // SNARK driver (e.g. `comm_eval_w`) to satisfy the trait's
    // `comm_eval` slot — which `IntegerModPCS::prove`/`verify` actually
    // ignore. The single value may be any F element, not bounded by
    // `T_f`. Skip limb-splitting in this case so a 128-bit Z_p eval
    // doesn't trip the bound check. Proper fix: drop `comm_eval` /
    // `blind_eval` from the trait (unused by IntegerModPCS). Tracked
    // in followups.
    //
    // TODO Phase 3 step D5: per-limb range check `|limb| < T`.
    let params = &ck.params;
    let v_limbs = if v.len() == 1 {
      v.to_vec()
    } else {
      limb_split_polynomial(v, params.log_t, params.log_t_f)
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
    ck_eval: &Self::CommitmentKey,
    transcript: &mut <T256DynPrimeEngine as SumcheckEngine>::TE,
    comm: &Self::Commitment,
    poly: &[BigUint],
    blind: &Self::Blind,
    point: &[<T256DynPrimeEngine as SumcheckEngine>::Scalar],
    eval: &BigUint,
    _comm_eval: &Self::Commitment,
    _blind_eval: &Self::Blind,
  ) -> Result<Self::EvaluationArgument, SpartanError> {
    let (_prove_span, prove_t) = start_span!("integer_modpcs_prove");
    let params = &ck.params;
    let monty = point
      .first()
      .map(|p| *p.params())
      .ok_or(SpartanError::InternalError {
        reason: "IntegerModPCS::prove: empty point".to_string(),
      })?;

    let (_red_span, red_t) = start_span!("imod_pcs_reduction");
    // 0. Limb-split f → f_limb. For numlimb=1 this is a literal pass-
    //    through; for numlimb>1 f_limb has 2^numlimb_var times as many
    //    coefficients (and 2^numlimb_var slots per original coefficient,
    //    padded with zero if numlimb isn't a power of two).
    let f_limb = limb_split_polynomial(poly, params.log_t, params.log_t_f);

    // 1. Reduction sumcheck (Phase-3 step D3): reduce the eval claim
    //    `f(int_r) ≡_p eval` to a claim about `f_limb` at a combined
    //    point `(int_r, r_k)` where `r_k` are the sumcheck challenges.
    //    Integrand: `limb(k) · f_limb(int_r, k)`, summed over k ∈
    //    {0,1}^numlimb_var. For numlimb_var = 0 the sumcheck has zero
    //    rounds, returns `r_k = []`, and the recovered eval equals
    //    the input `eval` directly (limb(empty) = T^0 = 1).
    let f_limb_p: Vec<crate::dyn_prime::DynPrime<4>> = f_limb
      .iter()
      .map(|b| {
        <crate::dyn_prime::DynPrime<4> as SumcheckField>::from_bytes_reduce(
          &monty,
          &b.to_bytes_le(),
        )
      })
      .collect();
    // Partial-eval f_limb at the original `point` in Z_p, leaving the
    // last numlimb_var variables free. Bind the top variables (= the
    // original n_vars) one at a time.
    let mut mle = crate::polys_modp::multilinear::MultilinearPolynomial::new(f_limb_p, monty);
    for r_i in point {
      mle.bind_poly_var_top(r_i);
    }
    let f_limb_at_int_r: Vec<crate::dyn_prime::DynPrime<4>> = mle.into_vec();
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
    let int_v_prime = integer_mle_evaluate(&f_limb, &int_point_ext);

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
    let poly_fq: Vec<t256::Scalar> = poly.iter().map(biguint_to_scalar).collect();
    info!(elapsed_ms = %red_t.elapsed().as_millis(), "imod_pcs_reduction");

    // 4. Phase 1: per prime, sample p_i, run all t iterations (if any),
    //    committing a_j_shifted / b_j_shifted and absorbing into the
    //    transcript. We stash the per-chain prover state needed to
    //    generate openings in phase 2.
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
    // oracles). Decoupling the primes from the per-chain commitments makes
    // the chains independent, so they're built — including the Hyrax
    // commits — in parallel. The commitments are then absorbed in order,
    // before γ, so the binding is unchanged.
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

    // Stack all (shifted) `a_j`/`b_j` into single commitments `F_a`/`F_b`.
    // Block `p(c,j)=c*t+(j-1)` holds the shifted poly in its low
    // `2^{n-jk}` slots, zero-padded to `MAX = 2^{n-k}`; blocks `p ≥ N` are
    // zero. `F_a`/`F_b` have size `n_pad·MAX` (a power of two). One Hyrax
    // commit + one absorb each, replacing the `2·s·t` per-iteration commits.
    let t = if with_iter {
      num_vars.saturating_sub(params.k).div_ceil(params.k)
    } else {
      0
    };
    let max_size = if with_iter {
      1usize << (num_vars - params.k)
    } else {
      0
    };
    let n_blocks = params.s * t;
    let n_pad = if n_blocks == 0 {
      0
    } else {
      n_blocks.next_power_of_two()
    };
    let stack_len = n_pad * max_size;

    let mut f_a_fq: Vec<t256::Scalar> = Vec::new();
    let mut f_b_fq: Vec<t256::Scalar> = Vec::new();
    let mut f_a_vals: Vec<BigUint> = Vec::new();
    let mut f_b_vals: Vec<BigUint> = Vec::new();
    let mut blind_f_a: Option<<Hyrax as PCSEngineTrait<T256HyraxEngine>>::Blind> = None;
    let mut blind_f_b: Option<<Hyrax as PCSEngineTrait<T256HyraxEngine>>::Blind> = None;
    let mut comm_f_a: Option<<Hyrax as PCSEngineTrait<T256HyraxEngine>>::Commitment> = None;
    let mut comm_f_b: Option<<Hyrax as PCSEngineTrait<T256HyraxEngine>>::Commitment> = None;
    if with_iter {
      let zero_u = BigUint::from(0u32);
      f_a_fq = vec![t256::Scalar::ZERO; stack_len];
      f_b_fq = vec![t256::Scalar::ZERO; stack_len];
      f_a_vals = vec![zero_u.clone(); stack_len];
      f_b_vals = vec![zero_u; stack_len];
      for (c, state) in chain_states.iter().enumerate() {
        for (jm1, it) in state.iters.iter().enumerate() {
          let base = stack_block(c, jm1 + 1, t) * max_size;
          let len = it.a_shifted_fq.len();
          f_a_fq[base..base + len].copy_from_slice(&it.a_shifted_fq);
          f_b_fq[base..base + len].copy_from_slice(&it.b_shifted_fq);
          f_a_vals[base..base + len].clone_from_slice(&it.a_shifted);
          f_b_vals[base..base + len].clone_from_slice(&it.b_shifted);
        }
      }
      let ba = Hyrax::blind(&ck.inner, stack_len);
      let bb = Hyrax::blind(&ck.inner, stack_len);
      let ca = Hyrax::commit(&ck.inner, &f_a_fq, &ba, false)?;
      let cb = Hyrax::commit(&ck.inner, &f_b_fq, &bb, false)?;
      transcript.absorb(b"f_a", &ca);
      transcript.absorb(b"f_b", &cb);
      blind_f_a = Some(ba);
      blind_f_b = Some(bb);
      comm_f_a = Some(ca);
      comm_f_b = Some(cb);
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

    // 6. Phase 2: per chain, generate openings. Borrow `chain_states`
    //    rather than consuming — D5.3 / D5.4 need to re-walk it after
    //    phase 2 to run deferred range checks on each iter's shifted
    //    polynomials.
    let (_open_span, open_t) = start_span!("imod_pcs_chain_openings");
    // Collect every chain-opening as a (point, eval) request against the
    // stacked `F_a` / `F_b` (or the input commitment `f` for the j=1 a_prev
    // and the t==0 final). The eval scalars also feed the identity/CRT
    // checks. Canonical order (mirrored by the verifier): per chain c, for
    // j=1..=t push [a_prev (j>1)] then [a_curr]; then push [final]; b-list
    // pushes [b_curr] per (c,j).
    let n = num_vars;
    let k = params.k;
    let log_np = if with_iter {
      n_pad.trailing_zeros() as usize
    } else {
      0
    };
    let max_log = if with_iter { num_vars - params.k } else { 0 };
    let mut a_points: Vec<Vec<t256::Scalar>> = Vec::new();
    let mut a_evals: Vec<t256::Scalar> = Vec::new();
    let mut b_points: Vec<Vec<t256::Scalar>> = Vec::new();
    let mut b_evals: Vec<t256::Scalar> = Vec::new();
    let mut input_points: Vec<Vec<t256::Scalar>> = Vec::with_capacity(params.s);
    let mut input_evals: Vec<t256::Scalar> = Vec::with_capacity(params.s);
    let mut chains: Vec<ChainData> = Vec::with_capacity(params.s);
    for (c, state) in chain_states.iter().enumerate() {
      let r_i_int = &state.r_i_int;
      let iters = &state.iters;
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

        // a_{j-1}(γ_ext): j=1 → input commitment (input batch); j>1 → the
        // a_{j-1} block of F_a (a-list).
        let a_prev_eval = if j == 1 {
          let e = mle_evaluate_fq(&poly_fq, &gamma_extended);
          input_points.push(gamma_extended.clone());
          input_evals.push(e);
          e
        } else {
          let prev = &iters[jm1 - 1];
          let e = mle_evaluate_fq(&prev.a_shifted_fq, &gamma_extended);
          a_points.push(stack_point(
            stack_block(c, j - 1, t),
            &gamma_extended,
            log_np,
            max_log,
          ));
          a_evals.push(e);
          e
        };

        // a_j(γ_prefix) and b_j(γ_prefix): curr opens against F_a / F_b.
        let a_curr_eval = mle_evaluate_fq(&iter_state.a_shifted_fq, &gamma_prefix);
        let b_curr_eval = mle_evaluate_fq(&iter_state.b_shifted_fq, &gamma_prefix);
        let p = stack_block(c, j, t);
        a_points.push(stack_point(p, &gamma_prefix, log_np, max_log));
        a_evals.push(a_curr_eval);
        b_points.push(stack_point(p, &gamma_prefix, log_np, max_log));
        b_evals.push(b_curr_eval);

        iter_oracles.push(IterationOracles {
          a_prev_eval,
          a_curr_eval,
          b_curr_eval,
        });
      }

      // Final-remainder eval: a_t at r_i[0..n-tk] (or f at r_i when t==0).
      let final_point_int: Vec<BigUint> = r_i_int[..(num_vars - t * params.k)].to_vec();
      let final_point_fq: Vec<t256::Scalar> =
        final_point_int.iter().map(biguint_to_scalar).collect();
      let final_eval = if t == 0 {
        let e = mle_evaluate_fq(&poly_fq, &final_point_fq);
        input_points.push(final_point_fq);
        input_evals.push(e);
        e
      } else {
        let last = &iters[t - 1];
        let e = mle_evaluate_fq(&last.a_shifted_fq, &final_point_fq);
        a_points.push(stack_point(
          stack_block(c, t, t),
          &final_point_fq,
          log_np,
          max_log,
        ));
        a_evals.push(e);
        e
      };

      chains.push(ChainData {
        iterations: iter_oracles,
        final_eval,
      });
    }
    info!(elapsed_ms = %open_t.elapsed().as_millis(), "imod_pcs_chain_openings");

    // F_a multi-point batch: bind all `a_j^c` opens (curr a_j@γ, a_prev
    // a_{j-1}@γ_ext for j>1, final a_t@r_i) to the single `F_a` commitment.
    let (_fa_span, fa_t) = start_span!("imod_pcs_f_a_batch");
    let f_a_batch = if with_iter {
      Some(prove_multipoint_batch(
        &ck.inner,
        &ck_eval.inner,
        transcript,
        b"f_a_batch",
        comm_f_a.as_ref().expect("F_a present when with_iter"),
        &f_a_fq,
        blind_f_a
          .as_ref()
          .expect("F_a blind present when with_iter"),
        &a_points,
        &a_evals,
      )?)
    } else {
      None
    };
    info!(elapsed_ms = %fa_t.elapsed().as_millis(), "imod_pcs_f_a_batch");

    // F_b multi-point batch: bind all `b_j^c` curr opens (b_j@γ) to `F_b`.
    let (_fb_span, fb_t) = start_span!("imod_pcs_f_b_batch");
    let f_b_batch = if with_iter {
      Some(prove_multipoint_batch(
        &ck.inner,
        &ck_eval.inner,
        transcript,
        b"f_b_batch",
        comm_f_b.as_ref().expect("F_b present when with_iter"),
        &f_b_fq,
        blind_f_b
          .as_ref()
          .expect("F_b blind present when with_iter"),
        &b_points,
        &b_evals,
      )?)
    } else {
      None
    };
    info!(elapsed_ms = %fb_t.elapsed().as_millis(), "imod_pcs_f_b_batch");

    // Input-commitment batch: the `j=1` a_prev opens (t>0) or the `s` final
    // opens (t==0), all against the input commitment `f`.
    let (_apb_span, apb_t) = start_span!("imod_pcs_aprev_batch");
    let a_prev_batch = if input_points.is_empty() {
      None
    } else {
      Some(prove_multipoint_batch(
        &ck.inner,
        &ck_eval.inner,
        transcript,
        b"aprev_batch",
        &comm.inner,
        &poly_fq,
        &blind.inner,
        &input_points,
        &input_evals,
      )?)
    };
    info!(elapsed_ms = %apb_t.elapsed().as_millis(), "imod_pcs_aprev_batch");

    // Phase-3 step D5 range checks: `f_limb` (input poly), then the two
    // stacked groups `F_a` and `F_b` (each range-checks every block of the
    // stack in one argument; all `a_j` share bound `2P`, all `b_j` share
    // `2q/P`). 3 groups for t>0, just `f_limb` for t==0.
    let log_bound_a = params.log_p + 1;
    let log_bound_b = LOG_Q - params.log_p + 1;
    let mut range_checks: Vec<BatchRangeCheck> = Vec::with_capacity(if with_iter { 3 } else { 1 });

    let (_rcf_span, rcf_t) = start_span!("imod_pcs_rc_flimb");
    // f_limb group (a single polynomial, bound `2^log_T`).
    range_checks.push(prove_batch_range_check(
      RangeBatchInputs {
        ck: &ck.inner,
        ck_eval: &ck_eval.inner,
        value_comm: &comm.inner,
        value_poly_fq: poly_fq.as_slice(),
        value_blind: &blind.inner,
        values: poly,
        n_pad: 1,
        n_values: poly.len(),
        log_bound: params.log_t,
      },
      transcript,
    )?);
    info!(elapsed_ms = %rcf_t.elapsed().as_millis(), "imod_pcs_rc_flimb");

    let (_rcab_span, rcab_t) = start_span!("imod_pcs_rc_ab");
    if with_iter {
      range_checks.push(prove_batch_range_check(
        RangeBatchInputs {
          ck: &ck.inner,
          ck_eval: &ck_eval.inner,
          value_comm: comm_f_a.as_ref().expect("F_a present when with_iter"),
          value_poly_fq: &f_a_fq,
          value_blind: blind_f_a
            .as_ref()
            .expect("F_a blind present when with_iter"),
          values: &f_a_vals,
          n_pad,
          n_values: max_size,
          log_bound: log_bound_a,
        },
        transcript,
      )?);
      range_checks.push(prove_batch_range_check(
        RangeBatchInputs {
          ck: &ck.inner,
          ck_eval: &ck_eval.inner,
          value_comm: comm_f_b.as_ref().expect("F_b present when with_iter"),
          value_poly_fq: &f_b_fq,
          value_blind: blind_f_b
            .as_ref()
            .expect("F_b blind present when with_iter"),
          values: &f_b_vals,
          n_pad,
          n_values: max_size,
          log_bound: log_bound_b,
        },
        transcript,
      )?);
    }
    info!(elapsed_ms = %rcab_t.elapsed().as_millis(), "imod_pcs_rc_ab");
    info!(elapsed_ms = %prove_t.elapsed().as_millis(), "integer_modpcs_prove");

    Ok(IntEvalArgument {
      reduction_round_polys,
      int_v_prime,
      chains,
      comm_f_a,
      comm_f_b,
      range_checks,
      f_a_batch,
      f_b_batch,
      a_prev_batch,
    })
  }

  fn verify(
    vk: &Self::VerifierKey,
    ck_eval: &Self::CommitmentKey,
    transcript: &mut <T256DynPrimeEngine as SumcheckEngine>::TE,
    comm: &Self::Commitment,
    point: &[<T256DynPrimeEngine as SumcheckEngine>::Scalar],
    eval: &BigUint,
    _comm_eval: &Self::Commitment,
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

    let (_vred_span, vred_t) = start_span!("imod_pcs_verify_reduction");
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
    info!(elapsed_ms = %vred_t.elapsed().as_millis(), "imod_pcs_verify_reduction");

    let (_vchain_span, vchain_t) = start_span!("imod_pcs_verify_chains");
    // 3. Phase 1: re-sample all `s` primes upfront, then absorb the chain
    //    commitments in order — mirroring the prover's reordered FS.
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
    // Stacked-commitment layout (mirror of the prover).
    let max_size = if with_iter {
      1usize << (num_vars - params.k)
    } else {
      0
    };
    let n_blocks = params.s * t;
    let n_pad = if n_blocks == 0 {
      0
    } else {
      n_blocks.next_power_of_two()
    };
    let log_np = if with_iter {
      n_pad.trailing_zeros() as usize
    } else {
      0
    };
    let max_log = if with_iter { num_vars - params.k } else { 0 };

    // Absorb the two stacked commitments (mirror of the prover), before γ.
    if with_iter {
      let ca = arg
        .comm_f_a
        .as_ref()
        .ok_or(SpartanError::InvalidSumcheckProof)?;
      let cb = arg
        .comm_f_b
        .as_ref()
        .ok_or(SpartanError::InvalidSumcheckProof)?;
      transcript.absorb(b"f_a", ca);
      transcript.absorb(b"f_b", cb);
    } else if arg.comm_f_a.is_some() || arg.comm_f_b.is_some() {
      return Err(SpartanError::InvalidSumcheckProof);
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

    // 5. Phase 2: per chain, verify each iteration's three openings +
    //    identity check, then verify the final-remainder opening + CRT.
    let shift_a_fq = biguint_to_scalar(&shift_a(params));
    let shift_b_fq = biguint_to_scalar(&shift_b(params));

    // Rebuild the F_a / F_b / input batch (point, eval) lists in the exact
    // same canonical order the prover used, and run the per-iteration
    // identity + final CRT checks on the sent eval scalars.
    let mut a_points: Vec<Vec<t256::Scalar>> = Vec::new();
    let mut a_evals: Vec<t256::Scalar> = Vec::new();
    let mut b_points: Vec<Vec<t256::Scalar>> = Vec::new();
    let mut b_evals: Vec<t256::Scalar> = Vec::new();
    let mut input_points: Vec<Vec<t256::Scalar>> = Vec::with_capacity(params.s);
    let mut input_evals: Vec<t256::Scalar> = Vec::with_capacity(params.s);
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

        // a_{j-1}(γ_ext): j=1 → input batch; j>1 → F_a a-list.
        if j == 1 {
          input_points.push(gamma_extended.clone());
          input_evals.push(iter.a_prev_eval);
        } else {
          a_points.push(stack_point(
            stack_block(chain_idx, j - 1, t),
            &gamma_extended,
            log_np,
            max_log,
          ));
          a_evals.push(iter.a_prev_eval);
        }
        // a_j(γ) and b_j(γ): curr opens against F_a / F_b.
        let p = stack_block(chain_idx, j, t);
        a_points.push(stack_point(p, &gamma_prefix, log_np, max_log));
        a_evals.push(iter.a_curr_eval);
        b_points.push(stack_point(p, &gamma_prefix, log_np, max_log));
        b_evals.push(iter.b_curr_eval);

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

      // Final-remainder eval: a_t at r_i[0..n-tk] (F_a) or f at r_i (input).
      let final_point_fq: Vec<t256::Scalar> = r_i_int[..(n - t * k)]
        .iter()
        .map(biguint_to_scalar)
        .collect();
      if t == 0 {
        input_points.push(final_point_fq);
        input_evals.push(chain.final_eval);
      } else {
        a_points.push(stack_point(
          stack_block(chain_idx, t, t),
          &final_point_fq,
          log_np,
          max_log,
        ));
        a_evals.push(chain.final_eval);
      }

      // CRT check: (final_eval [- shift_a if t>0]) as a *balanced* integer
      // ≡ int_v' (mod p_i).
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

    // Verify the F_a multi-point batch (all a opens) and the F_b batch (all
    // b curr opens), then the input batch (j=1 a_prev opens for t>0, or the
    // s final opens for t==0).
    let (_vfa_span, vfa_t) = start_span!("imod_pcs_verify_f_a_batch");
    if with_iter {
      let ca = arg
        .comm_f_a
        .as_ref()
        .ok_or(SpartanError::InvalidSumcheckProof)?;
      let batch = arg
        .f_a_batch
        .as_ref()
        .ok_or(SpartanError::InvalidSumcheckProof)?;
      verify_multipoint_batch(
        &vk.inner,
        &ck_eval.inner,
        transcript,
        b"f_a_batch",
        ca,
        (n_pad * max_size).trailing_zeros() as usize,
        &a_points,
        &a_evals,
        batch,
      )?;
      let cb = arg
        .comm_f_b
        .as_ref()
        .ok_or(SpartanError::InvalidSumcheckProof)?;
      let batch_b = arg
        .f_b_batch
        .as_ref()
        .ok_or(SpartanError::InvalidSumcheckProof)?;
      verify_multipoint_batch(
        &vk.inner,
        &ck_eval.inner,
        transcript,
        b"f_b_batch",
        cb,
        (n_pad * max_size).trailing_zeros() as usize,
        &b_points,
        &b_evals,
        batch_b,
      )?;
    } else if arg.f_a_batch.is_some() || arg.f_b_batch.is_some() {
      return Err(SpartanError::InvalidSumcheckProof);
    }
    info!(elapsed_ms = %vfa_t.elapsed().as_millis(), "imod_pcs_verify_f_a_batch");

    let (_vapb_span, vapb_t) = start_span!("imod_pcs_verify_aprev_batch");
    if input_points.is_empty() {
      if arg.a_prev_batch.is_some() {
        return Err(SpartanError::InvalidSumcheckProof);
      }
    } else {
      let batch = arg
        .a_prev_batch
        .as_ref()
        .ok_or(SpartanError::InvalidSumcheckProof)?;
      verify_multipoint_batch(
        &vk.inner,
        &ck_eval.inner,
        transcript,
        b"aprev_batch",
        &comm.inner,
        num_vars,
        &input_points,
        &input_evals,
        batch,
      )?;
    }
    info!(elapsed_ms = %vapb_t.elapsed().as_millis(), "imod_pcs_verify_aprev_batch");

    let (_vrc_span, vrc_t) = start_span!("imod_pcs_verify_rc");
    // Phase-3 step D5 range checks: f_limb (input), then the stacked F_a
    // and F_b groups. 3 groups for t>0, just f_limb for t==0.
    let expected_groups = if with_iter { 3 } else { 1 };
    if arg.range_checks.len() != expected_groups {
      return Err(SpartanError::InvalidSumcheckProof);
    }
    let log_bound_a = params.log_p + 1;
    let log_bound_b = LOG_Q - params.log_p + 1;

    // f_limb group (single polynomial, n_pad = 1).
    verify_batch_range_check(
      &vk.inner,
      &ck_eval.inner,
      &comm.inner,
      1,
      1usize << num_vars,
      params.log_t,
      &arg.range_checks[0],
      transcript,
    )?;

    // F_a / F_b groups: one stacked range check each.
    if with_iter {
      let ca = arg
        .comm_f_a
        .as_ref()
        .ok_or(SpartanError::InvalidSumcheckProof)?;
      verify_batch_range_check(
        &vk.inner,
        &ck_eval.inner,
        ca,
        n_pad,
        max_size,
        log_bound_a,
        &arg.range_checks[1],
        transcript,
      )?;
      let cb = arg
        .comm_f_b
        .as_ref()
        .ok_or(SpartanError::InvalidSumcheckProof)?;
      verify_batch_range_check(
        &vk.inner,
        &ck_eval.inner,
        cb,
        n_pad,
        max_size,
        log_bound_b,
        &arg.range_checks[2],
        transcript,
      )?;
    }
    info!(elapsed_ms = %vrc_t.elapsed().as_millis(), "imod_pcs_verify_rc");
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
/// serialized — holds the shifted `a_j` / `b_j` polynomials (integers and
/// F-cast) so they can be scattered into the stacked `F_a`/`F_b` buffers,
/// range-checked, and evaluated at the chain points. The commitments are
/// no longer per-iteration: `a_j`/`b_j` are blocks of the single `F_a`/`F_b`
/// commitments built after phase 1.
struct IterationProverState {
  /// `a_j_shifted` as integers (for the stacked range-check bit-decomposition).
  a_shifted: Vec<BigUint>,
  a_shifted_fq: Vec<t256::Scalar>,
  /// `b_j_shifted` as integers.
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
  let blind_eval = Hyrax::blind(ck_eval, 1);
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
    blind_eval,
    hyrax_arg: arg,
  })
}

/// Mirror of `hyrax_open_at` on the verifier side: reconstruct
/// `comm_eval` from the prover-sent `(f_y, blind_eval)` and verify the
/// Hyrax argument against the polynomial commitment `comm` at `point`.
fn hyrax_verify_open<T: ByteTranscript>(
  vk: &<Hyrax as PCSEngineTrait<T256HyraxEngine>>::VerifierKey,
  ck_eval: &<Hyrax as PCSEngineTrait<T256HyraxEngine>>::CommitmentKey,
  transcript: &mut T,
  comm: &<Hyrax as PCSEngineTrait<T256HyraxEngine>>::Commitment,
  point: &[t256::Scalar],
  opening: &SmallPrimeOpening,
) -> Result<(), SpartanError> {
  let comm_eval = Hyrax::commit(ck_eval, &[opening.f_y], &opening.blind_eval, false)?;
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

/// Inputs for a batch range check over a single (possibly pre-stacked)
/// value commitment. The value poly is `n_pad` blocks of `n_values`
/// coefficients each (`n_pad` a power of two; `1` for a lone poly), all
/// sharing the bound `2^log_bound`. Block `p` lives at
/// `[p·n_values .. p·n_values + n_values)`; padding blocks are zero. This
/// is exactly the stacked layout of `F_a`/`F_b`, so the value binding is a
/// single open of `value_comm` at `r_v` (no per-block fold).
struct RangeBatchInputs<'a> {
  /// The CK / VK pair (Hyrax inner).
  ck: &'a <Hyrax as PCSEngineTrait<T256HyraxEngine>>::CommitmentKey,
  ck_eval: &'a <Hyrax as PCSEngineTrait<T256HyraxEngine>>::CommitmentKey,
  /// The single value commitment, its F-cast coefficients (length
  /// `n_pad·n_values`), and blind.
  value_comm: &'a <Hyrax as PCSEngineTrait<T256HyraxEngine>>::Commitment,
  value_poly_fq: &'a [t256::Scalar],
  value_blind: &'a <Hyrax as PCSEngineTrait<T256HyraxEngine>>::Blind,
  /// Flat integer values, length `n_pad·n_values`; block `p` at
  /// `[p·n_values .. p·n_values + n_values)`. Padding blocks are zero.
  values: &'a [BigUint],
  /// Number of stacked blocks, padded to a power of two (`1` for a single poly).
  n_pad: usize,
  /// Coefficients per block (a power of two).
  n_values: usize,
  /// Bit-width of the shared bound (each value `< 2^log_bound`).
  log_bound: usize,
}

/// Masked power-of-two weight vector for the value-reconstruction
/// sumcheck: `w[b] = 2^b` for `b < log_bound`, else `0`. Length `stride`
/// (the padded per-value bit count). Bits at `b ≥ log_bound` carry zero
/// weight, so the prover can't inflate a value past its bound regardless
/// of those (still bit-valid) slots.
fn range_weight_vector(log_bound: usize, stride: usize) -> Vec<t256::Scalar> {
  let mut weight = Vec::with_capacity(stride);
  let mut pow = t256::Scalar::ONE;
  let two = t256::Scalar::from(2u64);
  for b in 0..stride {
    if b < log_bound {
      weight.push(pow);
      pow *= two;
    } else {
      weight.push(t256::Scalar::ZERO);
    }
  }
  weight
}

/// Spawn an F-side sub-transcript seeded from the parent, binding the
/// stacked bit commitment and the single value commitment. Both prover
/// and verifier reconstruct it identically.
fn spawn_range_subtranscript(
  parent: &mut Keccak256Transcript<T256DynPrimeEngine>,
  bit_comm: &<Hyrax as PCSEngineTrait<T256HyraxEngine>>::Commitment,
  value_comm: &<Hyrax as PCSEngineTrait<T256HyraxEngine>>::Commitment,
) -> Result<Keccak256Transcript<T256HyraxEngine>, SpartanError> {
  let seed = parent.squeeze_bytes(b"range_seed")?;
  let mut sub = <Keccak256Transcript<T256HyraxEngine> as TranscriptEngineTrait<
    T256HyraxEngine,
  >>::new_with_params(b"range_check", ());
  sub.absorb_bytes(b"seed", &seed);
  sub.absorb(b"range_bit_comm", bit_comm);
  sub.absorb(b"range_value_comm", value_comm);
  Ok(sub)
}

/// Spawn the sub-transcript for a multi-point batch evaluation, binding
/// the parent state, the committed polynomial `comm`, and the claimed
/// evals. `domain` separates the F_a / F_b / a_prev batches. The RLC
/// challenge `λ` is squeezed from this sub after the evals are bound. Both
/// prover and verifier reconstruct it identically.
fn spawn_batch_subtranscript(
  parent: &mut Keccak256Transcript<T256DynPrimeEngine>,
  domain: &'static [u8],
  comm: &<Hyrax as PCSEngineTrait<T256HyraxEngine>>::Commitment,
  evals: &[t256::Scalar],
) -> Result<Keccak256Transcript<T256HyraxEngine>, SpartanError> {
  let seed = parent.squeeze_bytes(domain)?;
  let mut sub = <Keccak256Transcript<T256HyraxEngine> as TranscriptEngineTrait<
    T256HyraxEngine,
  >>::new_with_params(domain, ());
  sub.absorb_bytes(b"seed", &seed);
  sub.absorb(b"mpb_comm", comm);
  for e in evals {
    sub.absorb_bytes(b"mpb_eval", e.to_repr().as_ref());
  }
  Ok(sub)
}

/// Build `W = Σ_i λ^i · eq(POINT_i, ·)` as length-`n` evals (the public
/// multi-point batch weight), and the combined claim `Σ_i λ^i · y_i`.
fn multipoint_weight(
  points: &[Vec<t256::Scalar>],
  evals: &[t256::Scalar],
  lambda: t256::Scalar,
  n: usize,
) -> (Vec<t256::Scalar>, t256::Scalar) {
  let mut w = vec![t256::Scalar::ZERO; n];
  let mut claim = t256::Scalar::ZERO;
  let mut lam_pow = t256::Scalar::ONE;
  for (z_c, &y_c) in points.iter().zip(evals.iter()) {
    let eq_c = EqPolynomial::<t256::Scalar>::evals_from_points(z_c);
    for (wj, &e) in w.iter_mut().zip(eq_c.iter()) {
      *wj += lam_pow * e;
    }
    claim += lam_pow * y_c;
    lam_pow *= lambda;
  }
  (w, claim)
}

/// Block index of chain `c`'s iteration `j` (`j ∈ [1,t]`) in the stacked
/// `F_a`/`F_b` layout. Chain-major, iteration-minor. MUST be identical on
/// prover and verifier.
#[inline]
fn stack_block(c: usize, j: usize, t: usize) -> usize {
  c * t + (j - 1)
}

/// Point addressing block `p`'s natural point `q` (length `≤ max_log`) in
/// the stacked index space of total width `log_np + max_log`:
/// `polyindex_bits(p) ++ [0; max_log − |q|] ++ q`, MSB-first (matching
/// `EqPolynomial`'s `r[0]`-is-high convention and Hyrax's row split). The
/// poly-index occupies the leading `log_np` coords; the zero pad pins the
/// within-block index below `2^|q|` (the live, non-padding region).
fn stack_point(p: usize, q: &[t256::Scalar], log_np: usize, max_log: usize) -> Vec<t256::Scalar> {
  let mut pt = Vec::with_capacity(log_np + max_log);
  for j in 0..log_np {
    let bit = (p >> (log_np - 1 - j)) & 1;
    pt.push(if bit == 1 {
      t256::Scalar::ONE
    } else {
      t256::Scalar::ZERO
    });
  }
  for _ in 0..(max_log - q.len()) {
    pt.push(t256::Scalar::ZERO);
  }
  pt.extend_from_slice(q);
  pt
}

/// Prover side of a multi-point batch evaluation of `poly_fq` (commitment
/// `comm`) at `points` with claimed `evals`: one degree-2 sumcheck on
/// `F·W` (`W = Σ_i λ^i eq(POINT_i)`) reducing to a single open of `F` at
/// the sumcheck challenge. Used for the F_a/F_b stacks and the input batch.
#[allow(clippy::too_many_arguments)]
fn prove_multipoint_batch(
  ck: &<Hyrax as PCSEngineTrait<T256HyraxEngine>>::CommitmentKey,
  ck_eval: &<Hyrax as PCSEngineTrait<T256HyraxEngine>>::CommitmentKey,
  parent: &mut Keccak256Transcript<T256DynPrimeEngine>,
  domain: &'static [u8],
  comm: &<Hyrax as PCSEngineTrait<T256HyraxEngine>>::Commitment,
  poly_fq: &[t256::Scalar],
  blind: &<Hyrax as PCSEngineTrait<T256HyraxEngine>>::Blind,
  points: &[Vec<t256::Scalar>],
  evals: &[t256::Scalar],
) -> Result<MultiPointBatch, SpartanError> {
  let n = poly_fq.len();
  let num_vars = n.trailing_zeros() as usize;
  let mut sub = spawn_batch_subtranscript(parent, domain, comm, evals)?;
  let lambda = sub.squeeze(b"mpb_lambda")?;
  let (_w_span, w_t) = start_span!("mpb_weight");
  let (w, claim) = multipoint_weight(points, evals, lambda, n);
  info!(elapsed_ms = %w_t.elapsed().as_millis(), npoints = points.len(), n = n, "mpb_weight");
  let mut poly_f = crate::polys::multilinear::MultilinearPolynomial::new(poly_fq.to_vec());
  let mut poly_w = crate::polys::multilinear::MultilinearPolynomial::new(w);
  let (_sc_span, sc_t) = start_span!("mpb_sumcheck");
  let (sumcheck, r, _claims) = crate::sumcheck::SumcheckProof::<T256HyraxEngine>::prove_quad(
    &claim,
    num_vars,
    &mut poly_f,
    &mut poly_w,
    &mut sub,
  )?;
  info!(elapsed_ms = %sc_t.elapsed().as_millis(), "mpb_sumcheck");
  let (_op_span, op_t) = start_span!("mpb_open");
  let f_open = hyrax_open_at(ck, ck_eval, &mut sub, comm, poly_fq, blind, &r)?;
  info!(elapsed_ms = %op_t.elapsed().as_millis(), "mpb_open");
  Ok(MultiPointBatch { sumcheck, f_open })
}

/// Verifier mirror of `prove_multipoint_batch`.
#[allow(clippy::too_many_arguments)]
fn verify_multipoint_batch(
  vk: &<Hyrax as PCSEngineTrait<T256HyraxEngine>>::VerifierKey,
  ck_eval: &<Hyrax as PCSEngineTrait<T256HyraxEngine>>::CommitmentKey,
  parent: &mut Keccak256Transcript<T256DynPrimeEngine>,
  domain: &'static [u8],
  comm: &<Hyrax as PCSEngineTrait<T256HyraxEngine>>::Commitment,
  num_vars: usize,
  points: &[Vec<t256::Scalar>],
  evals: &[t256::Scalar],
  batch: &MultiPointBatch,
) -> Result<(), SpartanError> {
  let mut sub = spawn_batch_subtranscript(parent, domain, comm, evals)?;
  let lambda = sub.squeeze(b"mpb_lambda")?;
  let mut claim = t256::Scalar::ZERO;
  let mut lam_pow = t256::Scalar::ONE;
  for &y_c in evals {
    claim += lam_pow * y_c;
    lam_pow *= lambda;
  }
  let (final_claim, r) = batch.sumcheck.verify(claim, num_vars, 2, &mut sub)?;
  hyrax_verify_open(vk, ck_eval, &mut sub, comm, &r, &batch.f_open)?;
  let mut w_at_r = t256::Scalar::ZERO;
  let mut lam_pow = t256::Scalar::ONE;
  for pt in points {
    w_at_r += lam_pow * EqPolynomial::<t256::Scalar>::new(pt.clone()).evaluate(&r);
    lam_pow *= lambda;
  }
  if final_claim != batch.f_open.f_y * w_at_r {
    return Err(SpartanError::InvalidSumcheckProof);
  }
  Ok(())
}

/// Prover side of a homogeneous batch range check (paper `rbatchrange`).
/// The `N` value polys are stacked along a top poly-index axis into one
/// bit polynomial of `N_pad · n_values · stride` bits (`N_pad =
/// next_pow2(N)`, `stride = 2^⌈log₂ log_bound⌉`), laid out as
/// `((p·n_values + within)·stride + b)`. One Hyrax commit, one bit-
/// validity zerocheck, one value-reconstruction sumcheck, two bit
/// openings, and one opening per value poly.
fn prove_batch_range_check(
  inputs: RangeBatchInputs<'_>,
  parent: &mut Keccak256Transcript<T256DynPrimeEngine>,
) -> Result<BatchRangeCheck, SpartanError> {
  let RangeBatchInputs {
    ck,
    ck_eval,
    value_comm,
    value_poly_fq,
    value_blind,
    values,
    n_pad,
    n_values,
    log_bound,
  } = inputs;

  debug_assert!(n_pad.is_power_of_two());
  debug_assert!(n_values.is_power_of_two());
  debug_assert_eq!(values.len(), n_pad * n_values);
  debug_assert_eq!(value_poly_fq.len(), n_pad * n_values);

  let log_np = n_pad.trailing_zeros() as usize;
  let log_nv = ceil_log2(n_values.max(1));
  let log_log_bound = ceil_log2(log_bound.max(1));
  let stride = 1usize << log_log_bound;
  let n_bits = n_pad * n_values * stride;
  let log_n_bits = ceil_log2(n_bits.max(1));

  // 1. Stacked bit polynomial. Index `((p·n_values + within)·stride + b)`.
  //    Bits `b ≥ log_bound` stay zero; padding-block values are already 0.
  //    Built in parallel over disjoint `stride`-sized slots (one per value);
  //    `gv = p·n_values + within` is the flat value index.
  let mut bit_poly: Vec<t256::Scalar> = vec![t256::Scalar::ZERO; n_bits];
  bit_poly
    .par_chunks_mut(stride)
    .enumerate()
    .for_each(|(gv, chunk)| {
      for (b, &bit) in bit_decompose_value(&values[gv], log_bound)
        .iter()
        .enumerate()
      {
        if bit == 1 {
          chunk[b] = t256::Scalar::ONE;
        }
      }
    });

  // 2. Commit the stacked bit polynomial.
  let bit_blind = Hyrax::blind(ck, n_bits);
  let bit_comm = Hyrax::commit(ck, &bit_poly, &bit_blind, true)?;

  // 3. Sub-transcript bound to (parent, bit_comm, value_comm).
  let mut sub = spawn_range_subtranscript(parent, &bit_comm, value_comm)?;

  // 4. Bit-validity zerocheck: `sum_x eq(x,τ)·(bit(x)² - bit(x)) = 0`.
  let tau: Vec<t256::Scalar> = (0..log_n_bits)
    .map(|_| sub.squeeze(b"range_tau"))
    .collect::<Result<Vec<_>, _>>()?;
  let mut poly_a = crate::polys::multilinear::MultilinearPolynomial::new(bit_poly.clone());
  let (bit_validity_sumcheck, r_validity, _claims) =
    crate::sumcheck::SumcheckProof::<T256HyraxEngine>::prove_cubic_square(
      &t256::Scalar::ZERO,
      tau,
      &mut poly_a,
      &mut sub,
    )?;

  // 5. Open bit_poly at r_validity.
  let bit_open_validity_final = hyrax_open_at(
    ck,
    ck_eval,
    &mut sub,
    &bit_comm,
    &bit_poly,
    &bit_blind,
    &r_validity,
  )?;

  // 6. Value-reconstruction. Squeeze r_v over (poly-index ++ within).
  let r_v: Vec<t256::Scalar> = (0..(log_np + log_nv))
    .map(|_| sub.squeeze(b"range_rv"))
    .collect::<Result<Vec<_>, _>>()?;
  // The value poly IS the stacked commitment, so its value at
  // `r_v = (r_v_poly ++ r_v_within)` is `V(r_v) = Σ_p eq(r_v_poly,p)·value_p(r_v_within)`
  // directly — a SINGLE open of `value_comm` at the full `r_v` (no fold).
  let value_open_at_rv = hyrax_open_at(
    ck,
    ck_eval,
    &mut sub,
    value_comm,
    value_poly_fq,
    value_blind,
    &r_v,
  )?;

  // Partial-eval bit_poly at r_v over the top (log_np + log_nv) vars,
  // leaving the b-axis (`stride` values).
  let mut bit_mle = crate::polys::multilinear::MultilinearPolynomial::new(bit_poly.clone());
  for r in &r_v {
    bit_mle.bind_poly_var_top(r);
  }
  let bit_at_rv: Vec<t256::Scalar> = bit_mle.into_vec();
  debug_assert_eq!(bit_at_rv.len(), stride);

  // Initial claim = V(r_v) = sum_b w[b]·bit_at_rv[b] (uniform weight).
  let weight = range_weight_vector(log_bound, stride);
  let claim_v: t256::Scalar = weight
    .iter()
    .zip(bit_at_rv.iter())
    .map(|(w, b)| *w * *b)
    .sum();
  let mut poly_w = crate::polys::multilinear::MultilinearPolynomial::new(weight);
  let mut poly_b2 = crate::polys::multilinear::MultilinearPolynomial::new(bit_at_rv);
  let (value_reconstr_sumcheck, r_b, _claims) =
    crate::sumcheck::SumcheckProof::<T256HyraxEngine>::prove_quad(
      &claim_v,
      log_log_bound,
      &mut poly_w,
      &mut poly_b2,
      &mut sub,
    )?;

  // 7. Open bit_poly at (r_v ++ r_b) for the reconstruction final check.
  let combined: Vec<t256::Scalar> = r_v.iter().chain(r_b.iter()).copied().collect();
  let bit_open_reconstr_final = hyrax_open_at(
    ck, ck_eval, &mut sub, &bit_comm, &bit_poly, &bit_blind, &combined,
  )?;

  Ok(BatchRangeCheck {
    bit_comm,
    bit_validity_sumcheck,
    value_reconstr_sumcheck,
    value_open_at_rv,
    bit_open_validity_final,
    bit_open_reconstr_final,
  })
}

/// Verifier-side mirror of `prove_batch_range_check`. Re-derives the
/// transcript challenges, re-runs the two sumchecks, and verifies the
/// three openings + the final integrand checks.
fn verify_batch_range_check(
  vk: &<Hyrax as PCSEngineTrait<T256HyraxEngine>>::VerifierKey,
  ck_eval: &<Hyrax as PCSEngineTrait<T256HyraxEngine>>::CommitmentKey,
  value_comm: &<Hyrax as PCSEngineTrait<T256HyraxEngine>>::Commitment,
  n_pad: usize,
  n_values: usize,
  log_bound: usize,
  arg: &BatchRangeCheck,
  parent: &mut Keccak256Transcript<T256DynPrimeEngine>,
) -> Result<(), SpartanError> {
  debug_assert!(n_pad.is_power_of_two());
  let log_np = n_pad.trailing_zeros() as usize;
  let log_nv = ceil_log2(n_values.max(1));
  let log_log_bound = ceil_log2(log_bound.max(1));
  let stride = 1usize << log_log_bound;
  let n_bits = n_pad * n_values * stride;
  let log_n_bits = ceil_log2(n_bits.max(1));

  // 1. Spawn the same sub-transcript the prover used.
  let mut sub = spawn_range_subtranscript(parent, &arg.bit_comm, value_comm)?;

  // 2. Bit-validity zerocheck (claim = 0, degree 3, log_n_bits rounds).
  let tau: Vec<t256::Scalar> = (0..log_n_bits)
    .map(|_| sub.squeeze(b"range_tau"))
    .collect::<Result<Vec<_>, _>>()?;
  let (bv_final_claim, r_validity) =
    arg
      .bit_validity_sumcheck
      .verify(t256::Scalar::ZERO, log_n_bits, 3, &mut sub)?;

  // 3. Verify bit_poly open at r_validity.
  hyrax_verify_open(
    vk,
    ck_eval,
    &mut sub,
    &arg.bit_comm,
    &r_validity,
    &arg.bit_open_validity_final,
  )?;

  // 4. Reconstruct bit-validity integrand: eq(r_validity, τ)·(bit² - bit).
  let eq_at_r = EqPolynomial::<t256::Scalar>::new(tau).evaluate(&r_validity);
  let bit_at_r = arg.bit_open_validity_final.f_y;
  let expected = eq_at_r * (bit_at_r * bit_at_r - bit_at_r);
  if bv_final_claim != expected {
    return Err(SpartanError::InvalidSumcheckProof);
  }

  // 5. Squeeze r_v = (poly-index ++ within) and verify the single open of
  //    the stacked value commitment at the full r_v. Its evaluation is
  //    V(r_v) = Σ_p eq(r_v_poly,p)·value_p(r_v_within) directly.
  let r_v: Vec<t256::Scalar> = (0..(log_np + log_nv))
    .map(|_| sub.squeeze(b"range_rv"))
    .collect::<Result<Vec<_>, _>>()?;
  hyrax_verify_open(
    vk,
    ck_eval,
    &mut sub,
    value_comm,
    &r_v,
    &arg.value_open_at_rv,
  )?;
  let value_at_rv = arg.value_open_at_rv.f_y;

  // 6. Value-reconstruction sumcheck (claim = V(r_v), degree 2,
  //    log_log_bound rounds).
  let (vr_final_claim, r_b) =
    arg
      .value_reconstr_sumcheck
      .verify(value_at_rv, log_log_bound, 2, &mut sub)?;

  // 7. Verify bit_poly open at (r_v ++ r_b).
  let combined: Vec<t256::Scalar> = r_v.iter().chain(r_b.iter()).copied().collect();
  hyrax_verify_open(
    vk,
    ck_eval,
    &mut sub,
    &arg.bit_comm,
    &combined,
    &arg.bit_open_reconstr_final,
  )?;

  // 8. Reconstruct integrand at r_b: w(r_b)·bit(r_v, r_b).
  let mut w_poly =
    crate::polys::multilinear::MultilinearPolynomial::new(range_weight_vector(log_bound, stride));
  for r in &r_b {
    w_poly.bind_poly_var_top(r);
  }
  let w_at_rb = w_poly.into_vec()[0];
  let bit_at_rv_rb = arg.bit_open_reconstr_final.f_y;
  let expected_vr = w_at_rb * bit_at_rv_rb;
  if vr_final_claim != expected_vr {
    return Err(SpartanError::InvalidSumcheckProof);
  }

  Ok(())
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

  /// `bit_decompose_value` is invertible by `sum_i 2^i · bit[i]`.
  #[test]
  fn bit_decompose_round_trips() {
    for (v, num_bits) in [
      (BigUint::from(0u32), 8),
      (BigUint::from(1u32), 1),
      (BigUint::from(0xffu32), 8),
      (BigUint::from(0xabcdu32), 16),
      (BigUint::from(0xdeadbeefu32), 32),
      (BigUint::from(0xffff_ffff_ffff_ffffu64), 64),
      (BigUint::from(0x7fff_ffffu32), 31), // odd bit count, top-bit-zero
    ] {
      let bits = bit_decompose_value(&v, num_bits);
      assert_eq!(bits.len(), num_bits);
      for b in &bits {
        assert!(*b == 0 || *b == 1);
      }
      let mut acc = BigUint::zero();
      for (i, b) in bits.iter().enumerate() {
        if *b == 1 {
          acc += BigUint::one() << i;
        }
      }
      assert_eq!(acc, v, "decomp of 0x{v:x} doesn't round-trip");
    }
  }

  /// `bit_decompose_polynomial` lays out `values.len() · num_bits`
  /// bits contiguously, with value `i`'s bits in slots `[i·num_bits,
  /// (i+1)·num_bits)`.
  #[test]
  fn bit_decompose_polynomial_layout() {
    let values = vec![
      BigUint::from(0b1010u32),
      BigUint::from(0b0011u32),
      BigUint::from(0b1100u32),
    ];
    let bits = bit_decompose_polynomial(&values, 4);
    assert_eq!(bits.len(), 12);
    // LE order: value[0] = 0b1010 = bits [0, 1, 0, 1]
    assert_eq!(&bits[0..4], &[0u8, 1, 0, 1]);
    assert_eq!(&bits[4..8], &[1u8, 1, 0, 0]);
    assert_eq!(&bits[8..12], &[0u8, 0, 1, 1]);
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
    let (ck_eval, _) = <MP as ModPCSEngineTrait<ME>>::setup(b"ck_eval", 1, 1);

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
    let comm = <MP as ModPCSEngineTrait<ME>>::commit(&ck, &poly, &blind, false).unwrap();
    let blind_eval = <MP as ModPCSEngineTrait<ME>>::blind(&ck_eval, 1);
    let comm_eval = <MP as ModPCSEngineTrait<ME>>::commit(
      &ck_eval,
      std::slice::from_ref(&eval),
      &blind_eval,
      false,
    )
    .unwrap();

    let mut pt = <ME as SumcheckEngine>::TE::new_with_params(b"intev", dyn_params);
    let arg = <MP as ModPCSEngineTrait<ME>>::prove(
      &ck,
      &ck_eval,
      &mut pt,
      &comm,
      &poly,
      &blind,
      &point,
      &eval,
      &comm_eval,
      &blind_eval,
    )
    .unwrap();

    let mut vt = <ME as SumcheckEngine>::TE::new_with_params(b"intev", dyn_params);
    <MP as ModPCSEngineTrait<ME>>::verify(
      &vk, &ck_eval, &mut vt, &comm, &point, &eval, &comm_eval, &arg,
    )
    .unwrap();
  }

  /// Verifier rejects a tampered claimed Z_p eval.
  #[test]
  fn verify_rejects_wrong_eval() {
    let num_vars = 4usize;
    let n = 1usize << num_vars;
    let (ck, vk) = <MP as ModPCSEngineTrait<ME>>::setup(b"inteval-rt", n, 256);
    let (ck_eval, _) = <MP as ModPCSEngineTrait<ME>>::setup(b"ck_eval", 1, 1);

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
    let comm = <MP as ModPCSEngineTrait<ME>>::commit(&ck, &poly, &blind, false).unwrap();
    let blind_eval = <MP as ModPCSEngineTrait<ME>>::blind(&ck_eval, 1);
    let comm_eval = <MP as ModPCSEngineTrait<ME>>::commit(
      &ck_eval,
      std::slice::from_ref(&real_eval),
      &blind_eval,
      false,
    )
    .unwrap();

    let mut pt = <ME as SumcheckEngine>::TE::new_with_params(b"intev", dyn_params);
    let arg = <MP as ModPCSEngineTrait<ME>>::prove(
      &ck,
      &ck_eval,
      &mut pt,
      &comm,
      &poly,
      &blind,
      &point,
      &real_eval,
      &comm_eval,
      &blind_eval,
    )
    .unwrap();

    // Verifier with the bad eval claim must reject.
    let mut vt = <ME as SumcheckEngine>::TE::new_with_params(b"intev", dyn_params);
    let err = <MP as ModPCSEngineTrait<ME>>::verify(
      &vk, &ck_eval, &mut vt, &comm, &point, &bad_eval, &comm_eval, &arg,
    )
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
    let (ck_eval, _) = <MP as ModPCSEngineTrait<ME>>::setup(b"ck_eval", 1, 1);

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
    let comm = <MP as ModPCSEngineTrait<ME>>::commit(&ck, &poly, &blind, false).unwrap();
    let blind_eval = <MP as ModPCSEngineTrait<ME>>::blind(&ck_eval, 1);
    let comm_eval = <MP as ModPCSEngineTrait<ME>>::commit(
      &ck_eval,
      std::slice::from_ref(&eval),
      &blind_eval,
      false,
    )
    .unwrap();

    let mut pt = <ME as SumcheckEngine>::TE::new_with_params(b"intev-iter", dyn_params);
    let arg = <MP as ModPCSEngineTrait<ME>>::prove(
      &ck,
      &ck_eval,
      &mut pt,
      &comm,
      &poly,
      &blind,
      &point,
      &eval,
      &comm_eval,
      &blind_eval,
    )
    .unwrap();

    let mut vt = <ME as SumcheckEngine>::TE::new_with_params(b"intev-iter", dyn_params);
    <MP as ModPCSEngineTrait<ME>>::verify(
      &vk, &ck_eval, &mut vt, &comm, &point, &eval, &comm_eval, &arg,
    )
    .unwrap();
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
    let (ck_eval, _) = <MP as ModPCSEngineTrait<ME>>::setup(b"ck_eval", 1, 1);

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
    let comm = <MP as ModPCSEngineTrait<ME>>::commit(&ck, &poly, &blind, false).unwrap();
    let blind_eval = <MP as ModPCSEngineTrait<ME>>::blind(&ck_eval, 1);
    let comm_eval = <MP as ModPCSEngineTrait<ME>>::commit(
      &ck_eval,
      std::slice::from_ref(&eval),
      &blind_eval,
      false,
    )
    .unwrap();

    let mut pt = <ME as SumcheckEngine>::TE::new_with_params(b"intev-iter2", dyn_params);
    let arg = <MP as ModPCSEngineTrait<ME>>::prove(
      &ck,
      &ck_eval,
      &mut pt,
      &comm,
      &poly,
      &blind,
      &point,
      &eval,
      &comm_eval,
      &blind_eval,
    )
    .unwrap();

    // Confirm we actually exercised t=2 with the stacked F_a/F_b design:
    // two iterations, both stacked commitments present, and the unified
    // a/b batches + the f_limb/F_a/F_b range groups emitted.
    assert_eq!(arg.chains[0].iterations.len(), 2, "expected t=2");
    assert!(
      arg.comm_f_a.is_some() && arg.comm_f_b.is_some(),
      "F_a/F_b present"
    );
    assert!(
      arg.f_a_batch.is_some() && arg.f_b_batch.is_some(),
      "a/b batches present"
    );
    assert!(
      arg.a_prev_batch.is_some(),
      "input (j=1 a_prev) batch present"
    );
    assert_eq!(arg.range_checks.len(), 3, "f_limb + F_a + F_b groups");

    let mut vt = <ME as SumcheckEngine>::TE::new_with_params(b"intev-iter2", dyn_params);
    <MP as ModPCSEngineTrait<ME>>::verify(
      &vk, &ck_eval, &mut vt, &comm, &point, &eval, &comm_eval, &arg,
    )
    .unwrap();
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
    let (ck_eval, _) = <MP as ModPCSEngineTrait<ME>>::setup(b"ck_eval", 1, 1);

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
    let comm = <MP as ModPCSEngineTrait<ME>>::commit(&ck, &poly, &blind, false).unwrap();
    let blind_eval = <MP as ModPCSEngineTrait<ME>>::blind(&ck_eval, 1);
    let comm_eval = <MP as ModPCSEngineTrait<ME>>::commit(
      &ck_eval,
      std::slice::from_ref(&eval),
      &blind_eval,
      false,
    )
    .unwrap();

    let mut pt = <ME as SumcheckEngine>::TE::new_with_params(b"intev-rc", dyn_params);
    let arg = <MP as ModPCSEngineTrait<ME>>::prove(
      &ck,
      &ck_eval,
      &mut pt,
      &comm,
      &poly,
      &blind,
      &point,
      &eval,
      &comm_eval,
      &blind_eval,
    )
    .unwrap();

    // Three groups (f_limb, a_1, b_1) — the config must produce them.
    assert_eq!(
      arg.range_checks.len(),
      3,
      "expected f_limb + a_1 + b_1 groups"
    );

    // Tampering each group's bit opening (in turn) must be rejected
    // (the Hyrax opening check or the bit-validity integrand check fires,
    // depending on which trips first).
    for gi in 0..arg.range_checks.len() {
      let mut bad = arg.clone();
      bad.range_checks[gi].bit_open_validity_final.f_y += t256::Scalar::ONE;
      let mut vt = <ME as SumcheckEngine>::TE::new_with_params(b"intev-rc", dyn_params);
      assert!(
        <MP as ModPCSEngineTrait<ME>>::verify(
          &vk, &ck_eval, &mut vt, &comm, &point, &eval, &comm_eval, &bad,
        )
        .is_err(),
        "group {gi} tamper not rejected"
      );
    }

    // Dropping a group (count mismatch) must also be rejected.
    let mut short = arg.clone();
    short.range_checks.pop();
    let mut vt = <ME as SumcheckEngine>::TE::new_with_params(b"intev-rc", dyn_params);
    assert!(
      <MP as ModPCSEngineTrait<ME>>::verify(
        &vk, &ck_eval, &mut vt, &comm, &point, &eval, &comm_eval, &short,
      )
      .is_err()
    );
  }

  /// Three-iteration roundtrip (`k=2`, `num_vars=8` → `t=3`). With `s·t`
  /// not a power of two the stacked `F_a`/`F_b` carry padding blocks
  /// (`n_pad > N`), exercising the zero-padded tail of the stack and the
  /// multi-point batch over the padded layout.
  #[test]
  fn prove_verify_roundtrips_with_three_iterations() {
    let num_vars = 8usize;
    let n = 1usize << num_vars;
    let small_params =
      IntEvalParams::derive_no_limb_split(8, 2, num_vars).expect("valid derived params");
    let (ck, vk) =
      IntegerModPCS::setup_with_params(b"inteval-iter3", n, 256, small_params).unwrap();
    let (ck_eval, _) = <MP as ModPCSEngineTrait<ME>>::setup(b"ck_eval", 1, 1);

    let dyn_params = small_dyn_params();
    // Keep coefficients < 2^log_t = 256 (the f_limb bound) at n = 256.
    let poly: Vec<BigUint> = (0..n)
      .map(|i| BigUint::from((i as u32 % 100) + 1))
      .collect();
    let point: Vec<DP> = (0..num_vars)
      .map(|i| DP::from_u64(&dyn_params, ((i as u64) * 3 + 5) % 37))
      .collect();
    let int_point: Vec<BigUint> = point.iter().map(dyn_to_biguint).collect();
    let eval = integer_mle_evaluate(&poly, &int_point)
      .mod_floor(&BigInt::from(37u32))
      .to_biguint()
      .unwrap();

    let blind = <MP as ModPCSEngineTrait<ME>>::blind(&ck, n);
    let comm = <MP as ModPCSEngineTrait<ME>>::commit(&ck, &poly, &blind, false).unwrap();
    let blind_eval = <MP as ModPCSEngineTrait<ME>>::blind(&ck_eval, 1);
    let comm_eval = <MP as ModPCSEngineTrait<ME>>::commit(
      &ck_eval,
      std::slice::from_ref(&eval),
      &blind_eval,
      false,
    )
    .unwrap();

    let mut pt = <ME as SumcheckEngine>::TE::new_with_params(b"intev-iter3", dyn_params);
    let arg = <MP as ModPCSEngineTrait<ME>>::prove(
      &ck,
      &ck_eval,
      &mut pt,
      &comm,
      &poly,
      &blind,
      &point,
      &eval,
      &comm_eval,
      &blind_eval,
    )
    .unwrap();
    assert_eq!(arg.chains[0].iterations.len(), 3, "expected t=3");

    let mut vt = <ME as SumcheckEngine>::TE::new_with_params(b"intev-iter3", dyn_params);
    <MP as ModPCSEngineTrait<ME>>::verify(
      &vk, &ck_eval, &mut vt, &comm, &point, &eval, &comm_eval, &arg,
    )
    .unwrap();
  }

  /// Multi-chain roundtrip + tamper coverage (`s=2`, `t=2`). Exercises the
  /// chain-major block mapping `p(c,j)=c·t+(j-1)` with `c>0`, and confirms
  /// every sent chain eval scalar is *bound* by the F_a/F_b batches:
  /// flipping `a_curr`, `b_curr`, a `j>1` `a_prev`, the `final` eval, or the
  /// F_a opening value must all reject (none is caught by the honest
  /// identity/CRT checks alone — only the multi-point batch binds them).
  #[test]
  fn verify_rejects_tampered_chain_evals() {
    let num_vars = 6usize;
    let n = 1usize << num_vars;
    // log_p=79 → Soundness-1 denom = 79-5-7-3 = 64, s = ⌈128/64⌉ = 2;
    // norms hold (k + (k+1)·log_p = 239 < 256). t = ⌈(6-2)/2⌉ = 2.
    let params = IntEvalParams::explicit(2, 79, 2, 8, 8, num_vars).expect("valid explicit params");
    assert_eq!(params.s, 2, "want a multi-chain config");
    let (ck, vk) = IntegerModPCS::setup_with_params(b"inteval-mc", n, 256, params).unwrap();
    let (ck_eval, _) = <MP as ModPCSEngineTrait<ME>>::setup(b"ck_eval", 1, 1);

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
    let comm = <MP as ModPCSEngineTrait<ME>>::commit(&ck, &poly, &blind, false).unwrap();
    let blind_eval = <MP as ModPCSEngineTrait<ME>>::blind(&ck_eval, 1);
    let comm_eval = <MP as ModPCSEngineTrait<ME>>::commit(
      &ck_eval,
      std::slice::from_ref(&eval),
      &blind_eval,
      false,
    )
    .unwrap();

    let mut pt = <ME as SumcheckEngine>::TE::new_with_params(b"intev-mc", dyn_params);
    let arg = <MP as ModPCSEngineTrait<ME>>::prove(
      &ck,
      &ck_eval,
      &mut pt,
      &comm,
      &poly,
      &blind,
      &point,
      &eval,
      &comm_eval,
      &blind_eval,
    )
    .unwrap();
    assert_eq!(arg.chains.len(), 2, "expected s=2 chains");
    assert_eq!(arg.chains[0].iterations.len(), 2, "expected t=2");

    // Honest roundtrip verifies.
    let reject = |bad: &IntEvalArgument| {
      let mut vt = <ME as SumcheckEngine>::TE::new_with_params(b"intev-mc", dyn_params);
      <MP as ModPCSEngineTrait<ME>>::verify(
        &vk, &ck_eval, &mut vt, &comm, &point, &eval, &comm_eval, bad,
      )
      .is_err()
    };
    {
      let mut vt = <ME as SumcheckEngine>::TE::new_with_params(b"intev-mc", dyn_params);
      <MP as ModPCSEngineTrait<ME>>::verify(
        &vk, &ck_eval, &mut vt, &comm, &point, &eval, &comm_eval, &arg,
      )
      .unwrap();
    }

    // Each tampered chain eval must be rejected by its multi-point batch.
    let one = t256::Scalar::ONE;
    {
      let mut bad = arg.clone();
      bad.chains[1].iterations[0].a_curr_eval += one;
      assert!(reject(&bad), "a_curr tamper not rejected");
    }
    {
      let mut bad = arg.clone();
      bad.chains[0].iterations[1].b_curr_eval += one;
      assert!(reject(&bad), "b_curr tamper not rejected");
    }
    {
      let mut bad = arg.clone();
      bad.chains[1].iterations[1].a_prev_eval += one;
      assert!(reject(&bad), "j>1 a_prev tamper not rejected");
    }
    {
      let mut bad = arg.clone();
      bad.chains[0].final_eval += one;
      assert!(reject(&bad), "final eval tamper not rejected");
    }
    {
      let mut bad = arg.clone();
      bad.f_a_batch.as_mut().unwrap().f_open.f_y += one;
      assert!(reject(&bad), "F_a open tamper not rejected");
    }
    {
      // Swapping F_a and F_b commitments must break both stacked range
      // checks / batches.
      let mut bad = arg.clone();
      std::mem::swap(&mut bad.comm_f_a, &mut bad.comm_f_b);
      assert!(reject(&bad), "F_a/F_b commitment swap not rejected");
    }
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
    let (ck_eval, _) = <MP as ModPCSEngineTrait<ME>>::setup(b"ck_eval", 1, 1);

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
    let comm = <MP as ModPCSEngineTrait<ME>>::commit(&ck, &poly, &blind, false).unwrap();
    let blind_eval = <MP as ModPCSEngineTrait<ME>>::blind(&ck_eval, 1);
    let comm_eval = <MP as ModPCSEngineTrait<ME>>::commit(
      &ck_eval,
      std::slice::from_ref(&eval),
      &blind_eval,
      false,
    )
    .unwrap();

    let mut pt = <ME as SumcheckEngine>::TE::new_with_params(b"limb-split", dyn_params);
    let arg = <MP as ModPCSEngineTrait<ME>>::prove(
      &ck,
      &ck_eval,
      &mut pt,
      &comm,
      &poly,
      &blind,
      &point,
      &eval,
      &comm_eval,
      &blind_eval,
    )
    .unwrap();
    // The reduction sumcheck ran one round → one entry in
    // reduction_round_polys.
    assert_eq!(arg.reduction_round_polys.len(), 1);

    let mut vt = <ME as SumcheckEngine>::TE::new_with_params(b"limb-split", dyn_params);
    <MP as ModPCSEngineTrait<ME>>::verify(
      &vk, &ck_eval, &mut vt, &comm, &point, &eval, &comm_eval, &arg,
    )
    .unwrap();
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
    let comm = <MP as ModPCSEngineTrait<ME>>::commit(&ck, &poly, &blind, false).unwrap();

    // Commit matches a direct Hyrax commit of the limb-split polynomial.
    let limbs = limb_split_polynomial(&poly, 4, 8);
    let limbs_fq: Vec<t256::Scalar> = limbs.iter().map(biguint_to_scalar).collect();
    let direct = Hyrax::commit(&ck.inner, &limbs_fq, &blind.inner, false).unwrap();
    assert_eq!(comm.inner, direct);
  }
}
