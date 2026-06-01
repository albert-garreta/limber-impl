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
  polys::eq::EqPolynomial,
  provider::{T256DynPrimeEngine, T256HyraxEngine, pcs::hyrax_pc::HyraxPCS, pt256::t256},
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

/// Evaluation argument: the prover-sent integer evaluation `int_v'` and
/// `s` small-prime openings. Step B targets the no-iteration (`n ≤ k`)
/// regime; step C will add partial-eval oracles.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct IntEvalEvalArg {
  /// `int_v' = f(int_r)` as a signed integer. Negative values come from
  /// `(1 - r_i)` factors in the multilinear chi. Serialized as
  /// `(sign, magnitude_le_bytes)`.
  pub int_v_prime: BigInt,
  /// One per small prime sampled from the transcript.
  pub openings: Vec<SmallPrimeOpening>,
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
/// once via `(q - 1) + 1` from `-Scalar::ONE`'s representation (same
/// trick as `bridge_modpcs::t256_scalar_params`); cheap enough to
/// recompute per call since it's just byte arithmetic.
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
    reason: "IntEvalModPCS: point must have at least one component to extract p".to_string(),
  })?;
  let modulus = p0.params().modulus();
  // `modulus` is `&Odd<Uint<4>>`; `.as_ref()` gives the inner `Uint<4>`.
  let bytes = modulus.as_ref().to_le_bytes();
  Ok(BigUint::from_bytes_le(bytes.as_slice()))
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
    let num_vars = point.len();
    if num_vars > params.k {
      // Phase 3 step C: partial-evaluation iteration. Not yet wired.
      return Err(SpartanError::InternalError {
        reason: format!(
          "IntEvalModPCS::prove: num_vars={num_vars} > k={}; partial-eval \
           iteration is Phase-3 step C (not yet implemented)",
          params.k
        ),
      });
    }

    // 1. Compute the signed integer evaluation `int_v' = f(int_r)`.
    let int_point: Vec<BigUint> = point.iter().map(dyn_to_biguint).collect();
    let int_v_prime = integer_mle_evaluate(poly, &int_point);

    // 2. Prover-side sanity: `eval ≡ int_v' (mod p)`.
    let p = extract_p(point)?;
    let int_v_mod_p = int_v_prime.mod_floor(&BigInt::from(p.clone()));
    let int_v_mod_p_u = int_v_mod_p
      .to_biguint()
      .expect("mod_floor of a BigInt by a positive BigUint is non-negative");
    if &int_v_mod_p_u != eval {
      return Err(SpartanError::InternalError {
        reason: "IntEvalModPCS::prove: eval ≠ int_v' mod p (prover bug)".to_string(),
      });
    }

    // 3. Bind `int_v'` into the transcript so verifier re-samples same primes.
    absorb_bigint(transcript, &int_v_prime);

    // 4. For each small prime `p_i`, compute `r_i = int_r mod p_i`,
    //    cast to F, and produce a Hyrax opening of `poly_fq` at `r_i_fq`.
    let poly_fq: Vec<t256::Scalar> = poly.iter().map(biguint_to_scalar).collect();
    let mut openings = Vec::with_capacity(params.s);
    for _ in 0..params.s {
      let p_i = sample_small_prime(transcript, params.log_p)?;
      let r_i_int: Vec<BigUint> = int_point.iter().map(|x| x % &p_i).collect();
      let r_i_fq: Vec<t256::Scalar> = r_i_int.iter().map(biguint_to_scalar).collect();

      let f_y = mle_evaluate_fq(&poly_fq, &r_i_fq);
      let blind_eval_i = Hyrax::blind(&ck_eval.inner, 1);
      let comm_eval_i = Hyrax::commit(&ck_eval.inner, &[f_y], &blind_eval_i, false)?;
      let hyrax_arg = Hyrax::prove(
        &ck.inner,
        &ck_eval.inner,
        transcript,
        &comm.inner,
        &poly_fq,
        &blind.inner,
        &r_i_fq,
        &comm_eval_i,
        &blind_eval_i,
      )?;

      openings.push(SmallPrimeOpening {
        f_y,
        blind_eval: blind_eval_i,
        hyrax_arg,
      });
    }

    Ok(IntEvalEvalArg {
      int_v_prime,
      openings,
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
    let num_vars = point.len();
    if num_vars > params.k {
      return Err(SpartanError::InternalError {
        reason: format!(
          "IntEvalModPCS::verify: num_vars={num_vars} > k={}; partial-eval \
           iteration is Phase-3 step C (not yet implemented)",
          params.k
        ),
      });
    }
    if arg.openings.len() != params.s {
      return Err(SpartanError::InvalidSumcheckProof);
    }

    // 1. Check `eval ≡ int_v' (mod p)`.
    let p = extract_p(point)?;
    let int_v_mod_p = arg.int_v_prime.mod_floor(&BigInt::from(p.clone()));
    let int_v_mod_p_u = int_v_mod_p
      .to_biguint()
      .ok_or(SpartanError::InvalidSumcheckProof)?;
    if &int_v_mod_p_u != eval {
      return Err(SpartanError::InvalidSumcheckProof);
    }

    // 2. Re-derive transcript binding identically to the prover.
    absorb_bigint(transcript, &arg.int_v_prime);

    // 3. Re-sample s small primes and verify each opening.
    let int_point: Vec<BigUint> = point.iter().map(dyn_to_biguint).collect();
    for opening in &arg.openings {
      let p_i = sample_small_prime(transcript, params.log_p)?;
      let r_i_int: Vec<BigUint> = int_point.iter().map(|x| x % &p_i).collect();
      let r_i_fq: Vec<t256::Scalar> = r_i_int.iter().map(biguint_to_scalar).collect();

      // Re-commit to `f_y` using the prover-sent blind; this binds
      // `comm_eval_i` to the specific `f_y` value the prover claims.
      let comm_eval_i = Hyrax::commit(&ck_eval.inner, &[opening.f_y], &opening.blind_eval, false)?;
      Hyrax::verify(
        &vk.inner,
        &ck_eval.inner,
        transcript,
        &comm.inner,
        &r_i_fq,
        &comm_eval_i,
        &opening.hyrax_arg,
      )?;

      // CRT congruence check: `to_int(f_y) ≡ int_v' (mod p_i)` in the
      // *balanced* F representation. `chi` factors include `(1 - r_i)`
      // which the F arithmetic computes as `q + 1 - r_i` — i.e. the
      // F value sits near `q` whenever the corresponding integer chi is
      // negative. Treating `f_y` as a signed integer in `[-q/2, q/2)`
      // recovers the integer value the integer MLE would produce; both
      // sides must agree mod p_i.
      let f_y_balanced = scalar_to_balanced_int(&opening.f_y);
      let lhs = f_y_balanced
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

  /// Prove rejects an `n > k` poly (Phase-3 step C boundary).
  #[test]
  fn prove_errors_above_k() {
    use crypto_bigint::{Odd, U256};
    let n = 1usize << 12; // 12 > k=7 (default)
    let (ck, _vk) = <MP as ModPCSEngineTrait<ME>>::setup(b"inteval-big", n, 256);
    let (ck_eval, _) = <MP as ModPCSEngineTrait<ME>>::setup(b"ck_eval", 1, 1);
    let dyn_params = small_dyn_params();
    let poly: Vec<BigUint> = (0..n).map(|i| BigUint::from(i as u32)).collect();
    let point: Vec<DP> = (0..12)
      .map(|i| DP::from_u64(&dyn_params, (i as u64) % 37))
      .collect();
    let blind = <MP as ModPCSEngineTrait<ME>>::blind(&ck, n);
    let comm = <MP as ModPCSEngineTrait<ME>>::commit(&ck, &poly, &blind, false).unwrap();
    let blind_eval = <MP as ModPCSEngineTrait<ME>>::blind(&ck_eval, 1);
    let eval = BigUint::from(0u32);
    let comm_eval = <MP as ModPCSEngineTrait<ME>>::commit(
      &ck_eval,
      std::slice::from_ref(&eval),
      &blind_eval,
      false,
    )
    .unwrap();
    let _ = (Odd::new(U256::from(3u32)), U256::ZERO);

    let mut pt = <ME as SumcheckEngine>::TE::new_with_params(b"intev", dyn_params);
    let err = <MP as ModPCSEngineTrait<ME>>::prove(
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
    .unwrap_err();
    assert!(matches!(err, SpartanError::InternalError { .. }));
  }
}
