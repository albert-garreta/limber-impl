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
  traits::{
    PrimeFieldExt,
    mod_engine::{ModPCSEngineTrait, SumcheckEngine, SumcheckField},
    pcs::PCSEngineTrait,
    transcript::{ByteTranscript, TranscriptReprTrait},
  },
};
use core::marker::PhantomData;
use ff::{Field, PrimeField};
use num_bigint::{BigInt, BigUint, Sign};
use num_integer::Integer;
use num_traits::{One, Zero};
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
  /// Opening of `a_{j-1}` at `(γ[0..n-jk], r^(i)[n-jk+1..n-(j-1)k])`.
  /// For `j=1` this opens the *input* polynomial's commitment, not an
  /// `IterationOracles` commitment.
  pub open_a_prev: SmallPrimeOpening,
  /// Opening of `a_j` at `γ[0..n-jk]`.
  pub open_a_curr: SmallPrimeOpening,
  /// Opening of `b_j` at `γ[0..n-jk]`.
  pub open_b_curr: SmallPrimeOpening,
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
  let mut out = Vec::with_capacity(values.len() * num_bits);
  for v in values {
    out.extend(bit_decompose_value(v, num_bits));
  }
  out
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
  let mut out = vec![BigUint::zero(); poly.len() * stride];
  for (x, v) in poly.iter().enumerate() {
    let limbs = split_value_into_limbs(v, log_t, numlimb);
    for (k, limb) in limbs.into_iter().enumerate() {
      out[x * stride + k] = limb;
    }
    // Slots [numlimb..stride) stay zero (padding when `numlimb` isn't a
    // power of two).
  }
  out
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

  let mut result = vec![BigInt::zero(); new_size];
  for (x, slot) in result.iter_mut().enumerate().take(new_size) {
    for (y, chi_y) in chi_table.iter().enumerate().take(two_k) {
      *slot += &poly[x * two_k + y] * chi_y;
    }
  }
  result
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
    let params = &ck.params;
    let monty = point
      .first()
      .map(|p| *p.params())
      .ok_or(SpartanError::InternalError {
        reason: "IntegerModPCS::prove: empty point".to_string(),
      })?;

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

    // 4. Phase 1: per prime, sample p_i, run all t iterations (if any),
    //    committing a_j_shifted / b_j_shifted and absorbing into the
    //    transcript. We stash the per-chain prover state needed to
    //    generate openings in phase 2.
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

        // a_prev starts as the input polynomial.
        let mut a_prev_int: Vec<BigInt> = poly.iter().map(|x| BigInt::from(x.clone())).collect();

        for j in 1..=t {
          let lo = n - j * k;
          let hi = n - (j - 1) * k;
          let r_lower = &r_i_int[lo..hi];

          let g_j_int = integer_partial_evaluate_top_k(&a_prev_int, r_lower);
          let mut a_j_int = Vec::with_capacity(g_j_int.len());
          let mut b_j_int = Vec::with_capacity(g_j_int.len());
          for g in &g_j_int {
            let (b, a) = truncated_divmod(g, &p_i);
            a_j_int.push(a);
            b_j_int.push(b);
          }

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
            a_shifted_fq: a_j_shifted_fq,
            a_blind: a_blind.clone(),
            comm_a_shifted: comm_a_shifted.clone(),
            b_shifted_fq: b_j_shifted_fq,
            b_blind,
            comm_b_shifted,
          });

          a_prev_int = a_j_int;
        }
      }

      chain_states.push(state);
    }

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

    // 6. Phase 2: per chain, generate openings.
    let mut chains: Vec<ChainData> = Vec::with_capacity(params.s);
    for state in chain_states {
      let ChainProverState {
        p_i: _,
        r_i_int,
        iters,
      } = state;

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

        let open_a_prev = hyrax_open_at(
          &ck.inner,
          &ck_eval.inner,
          transcript,
          &a_prev_comm,
          &a_prev_poly_fq,
          &a_prev_blind,
          &gamma_extended,
        )?;
        let open_a_curr = hyrax_open_at(
          &ck.inner,
          &ck_eval.inner,
          transcript,
          &iter_state.comm_a_shifted,
          &iter_state.a_shifted_fq,
          &iter_state.a_blind,
          &gamma_prefix,
        )?;
        let open_b_curr = hyrax_open_at(
          &ck.inner,
          &ck_eval.inner,
          transcript,
          &iter_state.comm_b_shifted,
          &iter_state.b_shifted_fq,
          &iter_state.b_blind,
          &gamma_prefix,
        )?;

        iter_oracles.push(IterationOracles {
          comm_a_shifted: iter_state.comm_a_shifted.clone(),
          comm_b_shifted: iter_state.comm_b_shifted.clone(),
          open_a_prev,
          open_a_curr,
          open_b_curr,
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

    Ok(IntEvalArgument {
      reduction_round_polys,
      int_v_prime,
      chains,
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
    let params = &vk.params;
    let monty = point
      .first()
      .map(|p| *p.params())
      .ok_or(SpartanError::InternalError {
        reason: "IntegerModPCS::verify: empty point".to_string(),
      })?;

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

        let a_prev_comm = if j == 1 {
          comm.inner.clone()
        } else {
          chain.iterations[jm1 - 1].comm_a_shifted.clone()
        };

        hyrax_verify_open(
          &vk.inner,
          &ck_eval.inner,
          transcript,
          &a_prev_comm,
          &gamma_extended,
          &iter.open_a_prev,
        )?;
        hyrax_verify_open(
          &vk.inner,
          &ck_eval.inner,
          transcript,
          &iter.comm_a_shifted,
          &gamma_prefix,
          &iter.open_a_curr,
        )?;
        hyrax_verify_open(
          &vk.inner,
          &ck_eval.inner,
          transcript,
          &iter.comm_b_shifted,
          &gamma_prefix,
          &iter.open_b_curr,
        )?;

        // Identity check in F: a_j(γ) + p_i · b_j(γ) ?= a_{j-1}(γ_ext).
        // a_prev for j=1 is the *unshifted* input poly (no subtract);
        // otherwise a_prev_shifted: subtract shift_a.
        let lhs_a = iter.open_a_curr.f_y - shift_a_fq;
        let lhs_b = iter.open_b_curr.f_y - shift_b_fq;
        let lhs = lhs_a + p_i_fq * lhs_b;
        let rhs = if j == 1 {
          iter.open_a_prev.f_y
        } else {
          iter.open_a_prev.f_y - shift_a_fq
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
  a_shifted_fq: Vec<t256::Scalar>,
  a_blind: <Hyrax as PCSEngineTrait<T256HyraxEngine>>::Blind,
  comm_a_shifted: <Hyrax as PCSEngineTrait<T256HyraxEngine>>::Commitment,
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

/// Helper: open the Hyrax commitment `comm` at `point` to produce a
/// `SmallPrimeOpening` (eval value + blind + Hyrax eval-argument). The
/// underlying polynomial `poly_fq` and its `blind` are inputs.
fn hyrax_open_at(
  ck: &<Hyrax as PCSEngineTrait<T256HyraxEngine>>::CommitmentKey,
  ck_eval: &<Hyrax as PCSEngineTrait<T256HyraxEngine>>::CommitmentKey,
  transcript: &mut Keccak256Transcript<T256DynPrimeEngine>,
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
fn hyrax_verify_open(
  vk: &<Hyrax as PCSEngineTrait<T256HyraxEngine>>::VerifierKey,
  ck_eval: &<Hyrax as PCSEngineTrait<T256HyraxEngine>>::CommitmentKey,
  transcript: &mut Keccak256Transcript<T256DynPrimeEngine>,
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
