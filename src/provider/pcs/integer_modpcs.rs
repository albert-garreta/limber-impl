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
    pcs::{FoldingEngineTrait, PCSEngineTrait},
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

/// One iteration's oracle commitments + identity-check openings at γ.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct IterationOracles {
  /// Hyrax commitment to the *shifted* `a_j` (each coefficient + `shift_a`).
  pub comm_a_shifted: <Hyrax as PCSEngineTrait<T256HyraxEngine>>::Commitment,
  /// Hyrax commitment to the *shifted* `b_j`.
  pub comm_b_shifted: <Hyrax as PCSEngineTrait<T256HyraxEngine>>::Commitment,
  /// Claimed `a_{j-1}(γ_ext)` where `γ_ext = (γ[0..n-jk], r^(i)[n-jk..n-(j-1)k])`.
  /// Used directly in the identity check.
  pub a_prev_eval: t256::Scalar,
  /// Opening binding `a_prev_eval` to the `a_{j-1}` commitment, at
  /// `γ_ext`. `None` for `j=1`: those open the shared *input* commitment
  /// at `s` distinct points and are batched across all chains into
  /// [`IntEvalArgument::a_prev_batch`]. `Some` for `j>1` (chain-specific
  /// commitment, one point — not batchable across chains).
  pub open_a_prev: Option<SmallPrimeOpening>,
  /// Claimed `a_j(γ[0..n-jk])`. Both `a_j` and `b_j` are opened at the
  /// same point, so instead of two Hyrax opens we send the two scalars
  /// (used directly in the identity check) plus a single `curr_open` of
  /// the RLC-folded commitment `comm_a + ρ·comm_b`.
  pub a_curr_eval: t256::Scalar,
  /// Claimed `b_j(γ[0..n-jk])`. See [`Self::a_curr_eval`].
  pub b_curr_eval: t256::Scalar,
  /// Opening of `comm_a_shifted + ρ·comm_b_shifted` at `γ[0..n-jk]`; its
  /// evaluation must equal `a_curr_eval + ρ·b_curr_eval` (binds both).
  pub curr_open: SmallPrimeOpening,
}

/// Per-prime chain: `t = ⌈(n-k)/k⌉` iterations plus the final-remainder
/// opening at `r^(i)[0..n-tk]`. For `n ≤ k` the iterations vec is empty
/// and `final_open` opens the input polynomial at `r^(i)`.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ChainData {
  /// Per-iteration oracle commitments and identity-check openings.
  pub iterations: Vec<IterationOracles>,
  /// `a_t(r^(i)[0..n-tk])` — opens the final remainder commitment for
  /// `t ≥ 1`; opens the input polynomial commitment for `t = 0`.
  pub final_open: SmallPrimeOpening,
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
  /// Phase-3 step D5 (stacked `rbatchrange`): one batched range check
  /// per `(bound, size)` group. Canonical group order is `f_limb`, then
  /// for each iteration `j = 1..=t` the `a_j` batch (all `s` chains) and
  /// the `b_j` batch. So `1 + 2t` entries (just `f_limb` when `t = 0`).
  /// Each batch range-checks all its polynomials with one bit
  /// commitment, one bit-validity zerocheck, and one reconstruction
  /// sumcheck. See [`prove_batch_range_check`].
  pub(crate) range_checks: Vec<BatchRangeCheck>,
  /// Batched proof for the `j=1` `a_prev` openings: all `s` chains open
  /// the shared input commitment at distinct points, collapsed into one
  /// sumcheck + one opening. `None` when there are no iterations (`t=0`).
  pub(crate) a_prev_batch: Option<APrevBatch>,
}

/// Multi-point batch evaluation of the input polynomial `f` (commitment
/// `comm.inner`) at the `s` distinct `j=1` `a_prev` points. Proves
/// `Σ_c λ^c·f(z_c) = Σ_x f(x)·W(x)` with `W = Σ_c λ^c·eq(z_c,·)` via one
/// degree-2 sumcheck reducing to a single opening of `f` at the
/// sumcheck challenge `r`. The claimed `f(z_c)` are the chains'
/// `iterations[0].a_prev_eval`.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct APrevBatch {
  /// Degree-2 sumcheck on `f(x)·W(x)`, over the Hyrax base field.
  pub(crate) sumcheck: crate::sumcheck::SumcheckProof<T256HyraxEngine>,
  /// Opening of the input commitment `f` at the sumcheck challenge `r`.
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
    IntegerModBlind {
      inner: Hyrax::blind(&ck.inner, n),
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
    let mut chain_states: Vec<ChainProverState> = Vec::with_capacity(params.s);
    for _ in 0..params.s {
      let p_i = sample_small_prime(transcript, params.log_p)?;
      let r_i_int: Vec<BigUint> = int_point.iter().map(|x| x % &p_i).collect();

      let mut state = ChainProverState {
        p_i: p_i.clone(),
        r_i_int: r_i_int.clone(),
        iters: Vec::new(),
      };

      if with_iter {
        let t = num_vars.saturating_sub(params.k).div_ceil(params.k);
        let n = num_vars;
        let k = params.k;

        // a_prev starts as the input polynomial (lifted once above).
        let mut a_prev_int: Vec<BigInt> = poly_bigint.clone();

        for j in 1..=t {
          let lo = n - j * k;
          let hi = n - (j - 1) * k;
          let r_lower = &r_i_int[lo..hi];

          let g_j_int = integer_partial_evaluate_top_k(&a_prev_int, r_lower);
          // Toward-zero divmod of every coefficient by p_i: `q · p_i + r = g`,
          // `(b, a) = (q, r)` — see `truncated_divmod`. `d_big` is hoisted
          // once per iteration instead of rebuilt per element.
          let d_big = BigInt::from(p_i.clone());
          let (b_j_int, a_j_int): (Vec<BigInt>, Vec<BigInt>) = g_j_int
            .par_iter()
            .map(|g| {
              let q = g / &d_big;
              let r = g - &q * &d_big;
              (q, r)
            })
            .unzip();

          let s_a = BigInt::from(shift_a(params));
          let s_b = BigInt::from(shift_b(params));
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

          let a_blind = Hyrax::blind(&ck.inner, a_j_shifted_fq.len());
          let b_blind = Hyrax::blind(&ck.inner, b_j_shifted_fq.len());
          let comm_a_shifted = Hyrax::commit(&ck.inner, &a_j_shifted_fq, &a_blind, false)?;
          let comm_b_shifted = Hyrax::commit(&ck.inner, &b_j_shifted_fq, &b_blind, false)?;

          transcript.absorb(b"a_shifted", &comm_a_shifted);
          transcript.absorb(b"b_shifted", &comm_b_shifted);

          state.iters.push(IterationProverState {
            a_shifted: a_j_shifted,
            a_shifted_fq: a_j_shifted_fq,
            a_blind: a_blind.clone(),
            comm_a_shifted: comm_a_shifted.clone(),
            b_shifted: b_j_shifted,
            b_shifted_fq: b_j_shifted_fq,
            b_blind,
            comm_b_shifted,
          });

          a_prev_int = a_j_int;
        }
      }

      chain_states.push(state);
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
    // `j=1` a_prev opens all hit the shared input commitment at distinct
    // points; collect them for the batched multi-point evaluation below.
    let mut aprev_points: Vec<Vec<t256::Scalar>> = Vec::with_capacity(params.s);
    let mut aprev_evals: Vec<t256::Scalar> = Vec::with_capacity(params.s);
    let mut chains: Vec<ChainData> = Vec::with_capacity(params.s);
    for state in &chain_states {
      let r_i_int = &state.r_i_int;
      let iters = &state.iters;

      // 6a. Generate identity-check openings for each iteration j.
      let mut iter_oracles = Vec::with_capacity(iters.len());
      let n = num_vars;
      let k = params.k;
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

        // a_{j-1}: for j=1 is the input commitment; otherwise iters[j-2]'s comm_a_shifted.
        let (a_prev_comm, a_prev_poly_fq, a_prev_blind) = if j == 1 {
          (comm.inner.clone(), poly_fq.clone(), blind.inner.clone())
        } else {
          let prev = &iters[jm1 - 1];
          (
            prev.comm_a_shifted.clone(),
            prev.a_shifted_fq.clone(),
            prev.a_blind.clone(),
          )
        };

        // a_prev eval (used in the identity check). For j=1 it opens the
        // shared input commitment → defer binding to the batch (collect
        // point+eval, no individual open). For j>1 open the chain-specific
        // commitment directly.
        let a_prev_eval = mle_evaluate_fq(&a_prev_poly_fq, &gamma_extended);
        let open_a_prev = if j == 1 {
          aprev_points.push(gamma_extended.clone());
          aprev_evals.push(a_prev_eval);
          None
        } else {
          Some(hyrax_open_at(
            &ck.inner,
            &ck_eval.inner,
            transcript,
            &a_prev_comm,
            &a_prev_poly_fq,
            &a_prev_blind,
            &gamma_extended,
          )?)
        };
        // a_j and b_j are both opened at `gamma_prefix`: send the two
        // claimed evals + ONE opening of `comm_a + ρ·comm_b`, saving an
        // IPA per iteration.
        let a_curr_eval = mle_evaluate_fq(&iter_state.a_shifted_fq, &gamma_prefix);
        let b_curr_eval = mle_evaluate_fq(&iter_state.b_shifted_fq, &gamma_prefix);
        let rho = squeeze_curr_rho(transcript, &a_curr_eval, &b_curr_eval)?;
        let one = t256::Scalar::ONE;
        let folded_comm = Hyrax::fold_commitments(
          &[
            iter_state.comm_a_shifted.clone(),
            iter_state.comm_b_shifted.clone(),
          ],
          &[one, rho],
        )?;
        let folded_blind = Hyrax::fold_blinds(
          &[iter_state.a_blind.clone(), iter_state.b_blind.clone()],
          &[one, rho],
        )?;
        let folded_poly: Vec<t256::Scalar> = iter_state
          .a_shifted_fq
          .iter()
          .zip(iter_state.b_shifted_fq.iter())
          .map(|(a, b)| *a + rho * *b)
          .collect();
        let curr_open = hyrax_open_at(
          &ck.inner,
          &ck_eval.inner,
          transcript,
          &folded_comm,
          &folded_poly,
          &folded_blind,
          &gamma_prefix,
        )?;

        iter_oracles.push(IterationOracles {
          comm_a_shifted: iter_state.comm_a_shifted.clone(),
          comm_b_shifted: iter_state.comm_b_shifted.clone(),
          a_prev_eval,
          open_a_prev,
          a_curr_eval,
          b_curr_eval,
          curr_open,
        });
      }

      // 6b. Final-remainder open: a_t at r_i[0..n-tk]. For t=0 this opens
      //     the *input* polynomial at the full r_i (step B path).
      let t = iters.len();
      let final_point_int: Vec<BigUint> = r_i_int[..(num_vars - t * params.k)].to_vec();
      let final_point_fq: Vec<t256::Scalar> =
        final_point_int.iter().map(biguint_to_scalar).collect();
      let final_open = if t == 0 {
        hyrax_open_at(
          &ck.inner,
          &ck_eval.inner,
          transcript,
          &comm.inner,
          &poly_fq,
          &blind.inner,
          &final_point_fq,
        )?
      } else {
        let last = &iters[t - 1];
        hyrax_open_at(
          &ck.inner,
          &ck_eval.inner,
          transcript,
          &last.comm_a_shifted,
          &last.a_shifted_fq,
          &last.a_blind,
          &final_point_fq,
        )?
      };

      chains.push(ChainData {
        iterations: iter_oracles,
        final_open,
      });
    }
    info!(elapsed_ms = %open_t.elapsed().as_millis(), "imod_pcs_chain_openings");

    // Batched `j=1` a_prev evaluation: prove all `s` claimed
    // `f(z_c) = aprev_evals[c]` against the shared input commitment with
    // one degree-2 sumcheck on `f(x)·W(x)` (W = Σ_c λ^c·eq(z_c,·)) plus a
    // single opening of `f` at the sumcheck challenge.
    let (_apb_span, apb_t) = start_span!("imod_pcs_aprev_batch");
    let a_prev_batch = if with_iter {
      let mut sub = spawn_aprev_subtranscript(transcript, &comm.inner, &aprev_evals)?;
      let lambda = sub.squeeze(b"aprev_lambda")?;
      let (w, claim) = aprev_batch_weight(&aprev_points, &aprev_evals, lambda, poly_fq.len());
      let mut poly_f = crate::polys::multilinear::MultilinearPolynomial::new(poly_fq.clone());
      let mut poly_w = crate::polys::multilinear::MultilinearPolynomial::new(w);
      let (sumcheck, r, _claims) = crate::sumcheck::SumcheckProof::<T256HyraxEngine>::prove_quad(
        &claim,
        num_vars,
        &mut poly_f,
        &mut poly_w,
        &mut sub,
      )?;
      let f_open = hyrax_open_at(
        &ck.inner,
        &ck_eval.inner,
        &mut sub,
        &comm.inner,
        &poly_fq,
        &blind.inner,
        &r,
      )?;
      Some(APrevBatch { sumcheck, f_open })
    } else {
      None
    };
    info!(elapsed_ms = %apb_t.elapsed().as_millis(), "imod_pcs_aprev_batch");

    // Phase-3 step D5 (stacked `rbatchrange`): one batched range check
    // per `(bound, size)` group, in canonical order `f_limb`, then for
    // each iteration `j` the `a_j` batch (all `s` chains) and the `b_j`
    // batch. Each batch folds `s` polynomials into a single argument.
    let t = if with_iter {
      num_vars.saturating_sub(params.k).div_ceil(params.k)
    } else {
      0
    };
    let log_bound_a = params.log_p + 1;
    let log_bound_b = LOG_Q - params.log_p + 1;
    let mut range_checks: Vec<BatchRangeCheck> = Vec::with_capacity(1 + 2 * t);

    let (_rcf_span, rcf_t) = start_span!("imod_pcs_rc_flimb");
    // f_limb group (a single polynomial, bound `2^log_T`).
    range_checks.push(prove_batch_range_check(
      RangeBatchInputs {
        ck: &ck.inner,
        ck_eval: &ck_eval.inner,
        value_comms: vec![&comm.inner],
        value_polys_fq: vec![poly_fq.as_slice()],
        value_blinds: vec![&blind.inner],
        values: vec![poly],
        n_values: poly.len(),
        log_bound: params.log_t,
      },
      transcript,
    )?);
    info!(elapsed_ms = %rcf_t.elapsed().as_millis(), "imod_pcs_rc_flimb");

    let (_rcab_span, rcab_t) = start_span!("imod_pcs_rc_ab");
    // For each iteration `j`, batch all `s` chains' `a_j` (bound `2P`)
    // then all `s` chains' `b_j` (bound `2q/P`). Same size per `j`.
    for j in 0..t {
      for (is_a, log_bound) in [(true, log_bound_a), (false, log_bound_b)] {
        let value_comms = chain_states
          .iter()
          .map(|st| {
            if is_a {
              &st.iters[j].comm_a_shifted
            } else {
              &st.iters[j].comm_b_shifted
            }
          })
          .collect::<Vec<_>>();
        let value_polys_fq = chain_states
          .iter()
          .map(|st| {
            if is_a {
              st.iters[j].a_shifted_fq.as_slice()
            } else {
              st.iters[j].b_shifted_fq.as_slice()
            }
          })
          .collect::<Vec<_>>();
        let value_blinds = chain_states
          .iter()
          .map(|st| {
            if is_a {
              &st.iters[j].a_blind
            } else {
              &st.iters[j].b_blind
            }
          })
          .collect::<Vec<_>>();
        let values = chain_states
          .iter()
          .map(|st| {
            if is_a {
              st.iters[j].a_shifted.as_slice()
            } else {
              st.iters[j].b_shifted.as_slice()
            }
          })
          .collect::<Vec<_>>();
        let n_values = values[0].len();
        range_checks.push(prove_batch_range_check(
          RangeBatchInputs {
            ck: &ck.inner,
            ck_eval: &ck_eval.inner,
            value_comms,
            value_polys_fq,
            value_blinds,
            values,
            n_values,
            log_bound,
          },
          transcript,
        )?);
      }
    }
    info!(elapsed_ms = %rcab_t.elapsed().as_millis(), "imod_pcs_rc_ab");
    info!(elapsed_ms = %prove_t.elapsed().as_millis(), "integer_modpcs_prove");

    Ok(IntEvalArgument {
      reduction_round_polys,
      int_v_prime,
      chains,
      range_checks,
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
    // 3. Phase 1: per chain, re-sample p_i, then for each iteration
    //    consume comm_a_shifted / comm_b_shifted from the arg and absorb
    //    them identically.
    let mut chain_primes: Vec<(BigUint, Vec<BigUint>)> = Vec::with_capacity(params.s);
    for chain in &arg.chains {
      let p_i = sample_small_prime(transcript, params.log_p)?;
      let r_i_int: Vec<BigUint> = int_point.iter().map(|x| x % &p_i).collect();

      if chain.iterations.len() != t {
        return Err(SpartanError::InvalidSumcheckProof);
      }
      for iter in &chain.iterations {
        transcript.absorb(b"a_shifted", &iter.comm_a_shifted);
        transcript.absorb(b"b_shifted", &iter.comm_b_shifted);
      }

      chain_primes.push((p_i, r_i_int));
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

    let mut aprev_points: Vec<Vec<t256::Scalar>> = Vec::with_capacity(params.s);
    let mut aprev_evals: Vec<t256::Scalar> = Vec::with_capacity(params.s);
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

        // a_prev: j=1 opens the shared input commitment → collect for the
        // batched verification below (no individual open). j>1 opens the
        // chain-specific commitment directly; check it binds a_prev_eval.
        if j == 1 {
          if iter.open_a_prev.is_some() {
            return Err(SpartanError::InvalidSumcheckProof);
          }
          aprev_points.push(gamma_extended.clone());
          aprev_evals.push(iter.a_prev_eval);
        } else {
          let a_prev_comm = chain.iterations[jm1 - 1].comm_a_shifted.clone();
          let open = iter
            .open_a_prev
            .as_ref()
            .ok_or(SpartanError::InvalidSumcheckProof)?;
          hyrax_verify_open(
            &vk.inner,
            &ck_eval.inner,
            transcript,
            &a_prev_comm,
            &gamma_extended,
            open,
          )?;
          if open.f_y != iter.a_prev_eval {
            return Err(SpartanError::InvalidSumcheckProof);
          }
        }
        // Batched a_j/b_j opening at `gamma_prefix`: re-derive ρ from the
        // claimed evals, fold `comm_a + ρ·comm_b`, verify the single open,
        // and check its eval equals `a_curr_eval + ρ·b_curr_eval`.
        let rho = squeeze_curr_rho(transcript, &iter.a_curr_eval, &iter.b_curr_eval)?;
        let folded_comm = Hyrax::fold_commitments(
          &[iter.comm_a_shifted.clone(), iter.comm_b_shifted.clone()],
          &[t256::Scalar::ONE, rho],
        )?;
        hyrax_verify_open(
          &vk.inner,
          &ck_eval.inner,
          transcript,
          &folded_comm,
          &gamma_prefix,
          &iter.curr_open,
        )?;
        if iter.curr_open.f_y != iter.a_curr_eval + rho * iter.b_curr_eval {
          return Err(SpartanError::InvalidSumcheckProof);
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

      // Final remainder verification.
      let final_point_fq: Vec<t256::Scalar> = r_i_int[..(n - t * k)]
        .iter()
        .map(biguint_to_scalar)
        .collect();
      let final_comm = if t == 0 {
        comm.inner.clone()
      } else {
        chain.iterations[t - 1].comm_a_shifted.clone()
      };
      hyrax_verify_open(
        &vk.inner,
        &ck_eval.inner,
        transcript,
        &final_comm,
        &final_point_fq,
        &chain.final_open,
      )?;

      // CRT check: (final_open.f_y [- shift_a if t>0]) interpreted as a
      // *balanced* integer ≡ int_v' (mod p_i).
      let final_f = if t == 0 {
        chain.final_open.f_y
      } else {
        chain.final_open.f_y - shift_a_fq
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

    // Verify the batched `j=1` a_prev evaluation: re-derive λ, reconstruct
    // the sumcheck claim `Σ_c λ^c·a_prev_eval`, verify the sumcheck + the
    // single `f` opening, and check `final_claim == f(r)·W(r)` with
    // `W(r) = Σ_c λ^c·eq(z_c, r)`.
    let (_vapb_span, vapb_t) = start_span!("imod_pcs_verify_aprev_batch");
    if with_iter {
      let batch = arg
        .a_prev_batch
        .as_ref()
        .ok_or(SpartanError::InvalidSumcheckProof)?;
      let mut sub = spawn_aprev_subtranscript(transcript, &comm.inner, &aprev_evals)?;
      let lambda = sub.squeeze(b"aprev_lambda")?;
      let mut claim = t256::Scalar::ZERO;
      let mut lam_pow = t256::Scalar::ONE;
      for &y_c in &aprev_evals {
        claim += lam_pow * y_c;
        lam_pow *= lambda;
      }
      let (final_claim, r) = batch.sumcheck.verify(claim, num_vars, 2, &mut sub)?;
      hyrax_verify_open(
        &vk.inner,
        &ck_eval.inner,
        &mut sub,
        &comm.inner,
        &r,
        &batch.f_open,
      )?;
      let mut w_at_r = t256::Scalar::ZERO;
      let mut lam_pow = t256::Scalar::ONE;
      for z_c in &aprev_points {
        w_at_r += lam_pow * EqPolynomial::<t256::Scalar>::new(z_c.clone()).evaluate(&r);
        lam_pow *= lambda;
      }
      if final_claim != batch.f_open.f_y * w_at_r {
        return Err(SpartanError::InvalidSumcheckProof);
      }
    } else if arg.a_prev_batch.is_some() {
      return Err(SpartanError::InvalidSumcheckProof);
    }
    info!(elapsed_ms = %vapb_t.elapsed().as_millis(), "imod_pcs_verify_aprev_batch");

    let (_vrc_span, vrc_t) = start_span!("imod_pcs_verify_rc");
    // Phase-3 step D5 (stacked rbatchrange): verify one batched range
    // check per (bound, size) group, in the same canonical order the
    // prover used: f_limb, then for each iteration j the a_j batch (all
    // s chains) and the b_j batch.
    let expected_groups = 1 + 2 * t;
    if arg.range_checks.len() != expected_groups {
      return Err(SpartanError::InvalidSumcheckProof);
    }
    let log_bound_a = params.log_p + 1;
    let log_bound_b = LOG_Q - params.log_p + 1;

    // f_limb group (single polynomial).
    verify_batch_range_check(
      &vk.inner,
      &ck_eval.inner,
      &[&comm.inner],
      1usize << num_vars,
      params.log_t,
      &arg.range_checks[0],
      transcript,
    )?;

    // a_j / b_j groups: all s chains' j-th iteration, same size per j.
    for j in 0..t {
      let n_values = 1usize << (num_vars - (j + 1) * params.k);
      for (offset, is_a, log_bound) in [(0usize, true, log_bound_a), (1usize, false, log_bound_b)] {
        let value_comms = arg
          .chains
          .iter()
          .map(|chain| {
            let it = &chain.iterations[j];
            if is_a {
              &it.comm_a_shifted
            } else {
              &it.comm_b_shifted
            }
          })
          .collect::<Vec<_>>();
        verify_batch_range_check(
          &vk.inner,
          &ck_eval.inner,
          &value_comms,
          n_values,
          log_bound,
          &arg.range_checks[1 + 2 * j + offset],
          transcript,
        )?;
      }
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
/// serialized — holds the underlying F polynomial / blind / commitment
/// for both `a_j_shifted` and `b_j_shifted` so phase 2 can produce
/// openings at γ.
struct IterationProverState {
  /// `a_j_shifted` as integers; kept so D5.3's deferred range check can
  /// re-bit-decompose without re-shifting / re-casting from F.
  a_shifted: Vec<BigUint>,
  a_shifted_fq: Vec<t256::Scalar>,
  a_blind: <Hyrax as PCSEngineTrait<T256HyraxEngine>>::Blind,
  comm_a_shifted: <Hyrax as PCSEngineTrait<T256HyraxEngine>>::Commitment,
  /// `b_j_shifted` as integers; kept so D5.4's deferred range check can
  /// re-bit-decompose without re-shifting / re-casting from F.
  b_shifted: Vec<BigUint>,
  b_shifted_fq: Vec<t256::Scalar>,
  b_blind: <Hyrax as PCSEngineTrait<T256HyraxEngine>>::Blind,
  comm_b_shifted: <Hyrax as PCSEngineTrait<T256HyraxEngine>>::Commitment,
}

/// Prover-side per-chain state collected in phase 1 and consumed in
/// phase 2.
struct ChainProverState {
  p_i: BigUint,
  r_i_int: Vec<BigUint>,
  iters: Vec<IterationProverState>,
}

/// Absorb the two claimed identity-check evals `a_j(γ)`, `b_j(γ)` into
/// the transcript and squeeze the RLC challenge `ρ`. The same `ρ` folds
/// `comm_a + ρ·comm_b` so a single opening at `γ` binds both evals (the
/// folded eval must equal `a_curr_eval + ρ·b_curr_eval`). Sampling `ρ`
/// after absorbing the evals is what makes the fold binding.
fn squeeze_curr_rho<T: ByteTranscript>(
  transcript: &mut T,
  a_eval: &t256::Scalar,
  b_eval: &t256::Scalar,
) -> Result<t256::Scalar, SpartanError> {
  let mut buf = a_eval.to_repr().as_ref().to_vec();
  buf.extend_from_slice(b_eval.to_repr().as_ref());
  transcript.absorb_bytes(b"curr_evals", &buf);
  let bytes = transcript.squeeze_bytes(b"curr_rho")?;
  Ok(<t256::Scalar as PrimeFieldExt>::from_uniform(&bytes))
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

/// Inputs for a homogeneous batch range check: `N` value polynomials,
/// all of length `n_values` (a power of two) and the same bound
/// `2^log_bound`. They are range-checked together as one `rbatchrange`.
struct RangeBatchInputs<'a> {
  /// The CK / VK pair (Hyrax inner).
  ck: &'a <Hyrax as PCSEngineTrait<T256HyraxEngine>>::CommitmentKey,
  ck_eval: &'a <Hyrax as PCSEngineTrait<T256HyraxEngine>>::CommitmentKey,
  /// One entry per polynomial, in the batch's canonical order: existing
  /// commitment, F-cast coefficients, and blind.
  value_comms: Vec<&'a <Hyrax as PCSEngineTrait<T256HyraxEngine>>::Commitment>,
  value_polys_fq: Vec<&'a [t256::Scalar]>,
  value_blinds: Vec<&'a <Hyrax as PCSEngineTrait<T256HyraxEngine>>::Blind>,
  values: Vec<&'a [BigUint]>,
  /// Coefficients per polynomial (same for all; a power of two).
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
/// stacked bit commitment and every value commitment in the batch. Both
/// prover and verifier reconstruct it identically.
fn spawn_range_subtranscript(
  parent: &mut Keccak256Transcript<T256DynPrimeEngine>,
  bit_comm: &<Hyrax as PCSEngineTrait<T256HyraxEngine>>::Commitment,
  value_comms: &[&<Hyrax as PCSEngineTrait<T256HyraxEngine>>::Commitment],
) -> Result<Keccak256Transcript<T256HyraxEngine>, SpartanError> {
  let seed = parent.squeeze_bytes(b"range_seed")?;
  let mut sub = <Keccak256Transcript<T256HyraxEngine> as TranscriptEngineTrait<
    T256HyraxEngine,
  >>::new_with_params(b"range_check", ());
  sub.absorb_bytes(b"seed", &seed);
  sub.absorb(b"range_bit_comm", bit_comm);
  for vc in value_comms {
    sub.absorb(b"range_value_comm", *vc);
  }
  Ok(sub)
}

/// Spawn the sub-transcript for the `j=1` `a_prev` batch evaluation,
/// binding the parent state, the input commitment `f`, and the claimed
/// evals `f(z_c)`. The RLC challenge `λ` is squeezed from this sub after
/// the evals are bound. Both prover and verifier reconstruct it identically.
fn spawn_aprev_subtranscript(
  parent: &mut Keccak256Transcript<T256DynPrimeEngine>,
  f_comm: &<Hyrax as PCSEngineTrait<T256HyraxEngine>>::Commitment,
  evals: &[t256::Scalar],
) -> Result<Keccak256Transcript<T256HyraxEngine>, SpartanError> {
  let seed = parent.squeeze_bytes(b"aprev_seed")?;
  let mut sub = <Keccak256Transcript<T256HyraxEngine> as TranscriptEngineTrait<
    T256HyraxEngine,
  >>::new_with_params(b"aprev_batch", ());
  sub.absorb_bytes(b"seed", &seed);
  sub.absorb(b"aprev_f_comm", f_comm);
  for e in evals {
    sub.absorb_bytes(b"aprev_eval", e.to_repr().as_ref());
  }
  Ok(sub)
}

/// Build `W = Σ_c λ^c · eq(z_c, ·)` as length-`n` evals (the public
/// multi-point batch weight), and the combined claim `Σ_c λ^c · y_c`.
/// Used by both the prover and verifier of the `a_prev` batch.
fn aprev_batch_weight(
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
    value_comms,
    value_polys_fq,
    value_blinds,
    values,
    n_values,
    log_bound,
  } = inputs;

  let num_polys = value_comms.len();
  debug_assert!(num_polys >= 1);
  debug_assert!(n_values.is_power_of_two());
  debug_assert!(values.iter().all(|v| v.len() == n_values));

  let n_pad = num_polys.next_power_of_two();
  let log_np = n_pad.trailing_zeros() as usize;
  let log_nv = ceil_log2(n_values.max(1));
  let log_log_bound = ceil_log2(log_bound.max(1));
  let stride = 1usize << log_log_bound;
  let n_bits = n_pad * n_values * stride;
  let log_n_bits = ceil_log2(n_bits.max(1));

  // 1. Stacked bit polynomial. Index `((p·n_values + within)·stride + b)`.
  //    Padding polys (`p ≥ num_polys`) and bits `b ≥ log_bound` stay zero.
  let mut bit_poly: Vec<t256::Scalar> = vec![t256::Scalar::ZERO; n_bits];
  for (p, vals) in values.iter().enumerate() {
    for (within, v) in vals.iter().enumerate() {
      let bits_u8 = bit_decompose_value(v, log_bound);
      let base = (p * n_values + within) * stride;
      for (b, &bit) in bits_u8.iter().enumerate() {
        if bit == 1 {
          bit_poly[base + b] = t256::Scalar::ONE;
        }
      }
    }
  }

  // 2. Commit the stacked bit polynomial.
  let bit_blind = Hyrax::blind(ck, n_bits);
  let bit_comm = Hyrax::commit(ck, &bit_poly, &bit_blind, true)?;

  // 3. Sub-transcript bound to (parent, bit_comm, all value_comms).
  let mut sub = spawn_range_subtranscript(parent, &bit_comm, &value_comms)?;

  // 4. Bit-validity zerocheck: `sum_x eq(x,τ)·(bit(x)² - bit(x)) = 0`.
  let tau: Vec<t256::Scalar> = (0..log_n_bits)
    .map(|_| sub.squeeze(b"range_tau"))
    .collect::<Result<Vec<_>, _>>()?;
  let mut poly_a = crate::polys::multilinear::MultilinearPolynomial::new(bit_poly.clone());
  let mut poly_b = crate::polys::multilinear::MultilinearPolynomial::new(bit_poly.clone());
  let mut poly_c = crate::polys::multilinear::MultilinearPolynomial::new(bit_poly.clone());
  let (bit_validity_sumcheck, r_validity, _claims) =
    crate::sumcheck::SumcheckProof::<T256HyraxEngine>::prove_cubic_with_three_inputs(
      &t256::Scalar::ZERO,
      tau,
      &mut poly_a,
      &mut poly_b,
      &mut poly_c,
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
  let r_v_poly = &r_v[..log_np];
  let r_v_within = &r_v[log_np..];

  // Fold the `num_polys` value polys/commitments/blinds by the weights
  // `eq(r_v_poly, p)` so a SINGLE Hyrax open yields
  // `V(r_v) = Σ_p eq(r_v_poly,p)·value_p(r_v_within)`. All polys share
  // `n_values`, so they share `r_v_within`; folding is exact by Pedersen
  // homomorphism. (For `num_polys = 1` the weight is 1 and this is the
  // plain single-poly open.)
  let eq_weights = EqPolynomial::<t256::Scalar>::new(r_v_poly.to_vec()).evals();
  let w = &eq_weights[..num_polys];
  let mut combined_poly = vec![t256::Scalar::ZERO; n_values];
  for (p, poly) in value_polys_fq.iter().enumerate() {
    for (o, &v) in combined_poly.iter_mut().zip(poly.iter()) {
      *o += w[p] * v;
    }
  }
  let comms_owned: Vec<_> = value_comms.iter().map(|c| (*c).clone()).collect();
  let blinds_owned: Vec<_> = value_blinds.iter().map(|b| (*b).clone()).collect();
  let combined_comm = Hyrax::fold_commitments(&comms_owned, w)?;
  let combined_blind = Hyrax::fold_blinds(&blinds_owned, w)?;
  let value_open_at_rv = hyrax_open_at(
    ck,
    ck_eval,
    &mut sub,
    &combined_comm,
    &combined_poly,
    &combined_blind,
    r_v_within,
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
  value_comms: &[&<Hyrax as PCSEngineTrait<T256HyraxEngine>>::Commitment],
  n_values: usize,
  log_bound: usize,
  arg: &BatchRangeCheck,
  parent: &mut Keccak256Transcript<T256DynPrimeEngine>,
) -> Result<(), SpartanError> {
  let num_polys = value_comms.len();
  if num_polys == 0 {
    return Err(SpartanError::InvalidSumcheckProof);
  }
  let n_pad = num_polys.next_power_of_two();
  let log_np = n_pad.trailing_zeros() as usize;
  let log_nv = ceil_log2(n_values.max(1));
  let log_log_bound = ceil_log2(log_bound.max(1));
  let stride = 1usize << log_log_bound;
  let n_bits = n_pad * n_values * stride;
  let log_n_bits = ceil_log2(n_bits.max(1));

  // 1. Spawn the same sub-transcript the prover used.
  let mut sub = spawn_range_subtranscript(parent, &arg.bit_comm, value_comms)?;

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

  // 5. Squeeze r_v = (poly-index ++ within), fold the value commitments
  //    by `eq(r_v_poly, p)` (matching the prover), and verify the single
  //    folded open. Its evaluation is V(r_v) = Σ_p eq(r_v_poly,p)·value_p.
  let r_v: Vec<t256::Scalar> = (0..(log_np + log_nv))
    .map(|_| sub.squeeze(b"range_rv"))
    .collect::<Result<Vec<_>, _>>()?;
  let r_v_poly = &r_v[..log_np];
  let r_v_within = &r_v[log_np..];
  let eq_weights = EqPolynomial::<t256::Scalar>::new(r_v_poly.to_vec()).evals();
  let w = &eq_weights[..num_polys];
  let comms_owned: Vec<_> = value_comms.iter().map(|c| (*c).clone()).collect();
  let combined_comm = Hyrax::fold_commitments(&comms_owned, w)?;
  hyrax_verify_open(
    vk,
    ck_eval,
    &mut sub,
    &combined_comm,
    r_v_within,
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

    // Confirm we actually exercised t=2 (j=1 batched + j=2 individual).
    assert_eq!(arg.chains[0].iterations.len(), 2, "expected t=2");
    assert!(
      arg.chains[0].iterations[0].open_a_prev.is_none(),
      "j=1 a_prev is batched"
    );
    assert!(
      arg.chains[0].iterations[1].open_a_prev.is_some(),
      "j=2 a_prev is individual"
    );

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
}
