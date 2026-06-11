//! LogUp-GKR range proof.
//!
//! Proves that every value in a witness vector lies in `[0, 2^bits)` using the
//! logarithmic-derivative (LogUp) lookup identity
//!
//! ```text
//!   sum_{i=0}^{N-1} 1/(r + w[i])  =  sum_{j=0}^{2^bits - 1} m_j/(r + j)
//! ```
//!
//! where `m_j` is the multiplicity (number of witnesses equal to `j`). The
//! identity holds for a random challenge `r` iff the multiset `{w[i]}` is
//! contained in the table `[0, 2^bits)` — which is exactly the range claim.
//!
//! Each side of the identity is a sum of rational functions, evaluated by a
//! layered GKR circuit whose gates add fractions:
//!
//! ```text
//!   (p_l, q_l) + (p_r, q_r) = (p_l·q_r + p_r·q_l,  q_l·q_r)
//! ```
//!
//! A balanced binary tree of these gates reduces `2^d` leaf fractions to a
//! single root fraction `(P, Q) = (numerator, denominator)` equal to the whole
//! sum. We run one tree per side and check the cross-product
//! `P_L·Q_R == P_R·Q_L` (i.e. `P_L/Q_L == P_R/Q_R`).
//!
//! The GKR protocol reduces the root claim, layer by layer, to a single
//! evaluation claim about each side's *leaf* multilinear extension at a random
//! point. For the witness side that claim pins `w(ρ_L)`; for the table side it
//! pins `m(ρ_R)`. Those two evaluations are returned as [`RangeClaims`] so the
//! caller can discharge them with PCS openings of the committed witness and
//! multiplicity polynomials. The table-index and all-ones-numerator leaves are
//! structured, so the verifier checks them in closed form.
//!
//! This module is intentionally self-contained (no PCS dependency yet) so the
//! GKR core and the LogUp identity can be tested in isolation before wiring the
//! returned [`RangeClaims`] into a commitment scheme. **Fiat–Shamir note:** a
//! real caller must absorb the witness and multiplicity commitments into the
//! transcript *before* calling [`LogUpRangeProof::prove`]/`verify`, so the
//! challenge `r` is bound to them.

use crate::{
  errors::SpartanError,
  polys::eq::EqPolynomial,
  traits::{
    Engine,
    transcript::{ByteTranscript, TranscriptEngineTrait},
  },
};
use ff::{Field, PrimeField};
use rayon::prelude::*;
use serde::{Deserialize, Serialize};

/// Below this many index pairs, the hot loops run serially (rayon overhead
/// would dominate).
const PAR_THRESHOLD: usize = 1 << 12;

/// In-place bind of a dense MLE's top variable to `r`: replaces the table with
/// its restriction `Z(r, ·)` (length halves). Matches the big-endian
/// convention of [`crate::polys::multilinear::MultilinearPolynomial`].
#[inline]
fn bind_top<F: PrimeField>(v: &mut Vec<F>, r: F) {
  let h = v.len() / 2;
  if h >= PAR_THRESHOLD {
    let (lo, hi) = v.split_at_mut(h);
    lo.par_iter_mut().zip(hi.par_iter()).for_each(|(a, b)| {
      *a += r * (*b - *a);
    });
  } else {
    for i in 0..h {
      let diff = v[i + h] - v[i];
      v[i] += r * diff;
    }
  }
  v.truncate(h);
}

/// Evaluate a degree-3 univariate at `r` from its evaluations at `0,1,2,3`
/// via Lagrange interpolation.
fn eval_cubic<F: PrimeField>(e: &[F; 4], r: F) -> F {
  let inv2 = F::from(2).invert().expect("2 invertible");
  let inv6 = F::from(6).invert().expect("6 invertible");
  let r1 = r - F::ONE;
  let r2 = r - F::from(2);
  let r3 = r - F::from(3);
  // Lagrange basis at nodes {0,1,2,3}.
  let l0 = r1 * r2 * r3 * (-inv6);
  let l1 = r * r2 * r3 * inv2;
  let l2 = r * r1 * r3 * (-inv2);
  let l3 = r * r1 * r2 * inv6;
  e[0] * l0 + e[1] * l1 + e[2] * l2 + e[3] * l3
}

/// Closed-form evaluation of the MLE of the table-index function
/// `f(j) = j` over `{0,1}^len`, at `point` (with `point[0]` the most
/// significant variable): `sum_k point[k]·2^(len-1-k)`.
fn idx_mle_eval<F: PrimeField>(point: &[F]) -> F {
  let mut acc = F::ZERO;
  let mut pow = F::ONE;
  for k in (0..point.len()).rev() {
    acc += point[k] * pow;
    pow = pow + pow;
  }
  acc
}

/// One GKR layer's reduction: the cubic-sumcheck round polynomials (each as
/// evaluations at `0,1,2,3`) plus the input layer's four evaluations
/// `(p(0,ρ'), p(1,ρ'), q(0,ρ'), q(1,ρ'))` at the sumcheck point.
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(bound = "")]
pub(crate) struct GkrLayerProof<E: Engine> {
  round_polys: Vec<[E::Scalar; 4]>,
  p0: E::Scalar,
  p1: E::Scalar,
  q0: E::Scalar,
  q1: E::Scalar,
}

/// A fractional-sum GKR proof: one [`GkrLayerProof`] per layer, top (root) to
/// bottom (leaves).
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(bound = "")]
pub(crate) struct GkrProof<E: Engine> {
  layers: Vec<GkrLayerProof<E>>,
}

/// Prover output for one fraction tree.
struct GkrOut<E: Engine> {
  proof: GkrProof<E>,
  root_p: E::Scalar,
  root_q: E::Scalar,
  leaf_point: Vec<E::Scalar>,
  leaf_p: E::Scalar,
  leaf_q: E::Scalar,
}

/// Build all layers of the fraction tree from the leaves up to the root.
/// `levels_p[k]` / `levels_q[k]` hold the `2^k`-entry layer; `[d]` is the
/// leaves and `[0]` the single-entry root.
fn build_levels<E: Engine>(
  p_leaves: Vec<E::Scalar>,
  q_leaves: Vec<E::Scalar>,
) -> (Vec<Vec<E::Scalar>>, Vec<Vec<E::Scalar>>) {
  let n = p_leaves.len();
  let d = n.trailing_zeros() as usize;
  let mut levels_p = vec![Vec::new(); d + 1];
  let mut levels_q = vec![Vec::new(); d + 1];
  levels_p[d] = p_leaves;
  levels_q[d] = q_leaves;
  for k in (0..d).rev() {
    let half = 1usize << k;
    let mut op = vec![E::Scalar::ZERO; half];
    let mut oq = vec![E::Scalar::ZERO; half];
    {
      let in_p = &levels_p[k + 1];
      let in_q = &levels_q[k + 1];
      // Combine the (top=0, i) and (top=1, i) fractions.
      if half >= PAR_THRESHOLD {
        op.par_iter_mut()
          .zip(oq.par_iter_mut())
          .enumerate()
          .for_each(|(i, (p, q))| {
            *p = in_p[i] * in_q[i + half] + in_p[i + half] * in_q[i];
            *q = in_q[i] * in_q[i + half];
          });
      } else {
        for i in 0..half {
          op[i] = in_p[i] * in_q[i + half] + in_p[i + half] * in_q[i];
          oq[i] = in_q[i] * in_q[i + half];
        }
      }
    }
    levels_p[k] = op;
    levels_q[k] = oq;
  }
  (levels_p, levels_q)
}

/// Prove that `(root_p, root_q)` is the sum of the leaf fractions
/// `(p_leaves[i], q_leaves[i])`, reducing the root claim to a single
/// evaluation claim on the leaf MLEs.
fn gkr_prove<E: Engine>(
  p_leaves: Vec<E::Scalar>,
  q_leaves: Vec<E::Scalar>,
  transcript: &mut E::TE,
) -> Result<GkrOut<E>, SpartanError> {
  let n = p_leaves.len();
  assert!(n.is_power_of_two() && n == q_leaves.len() && n >= 1);
  let d = n.trailing_zeros() as usize;
  let (levels_p, levels_q) = build_levels::<E>(p_leaves, q_leaves);

  let root_p = levels_p[0][0];
  let root_q = levels_q[0][0];
  transcript.absorb(b"gkr_root_p", &root_p);
  transcript.absorb(b"gkr_root_q", &root_q);

  let mut layers = Vec::with_capacity(d);
  let mut point: Vec<E::Scalar> = Vec::new();
  let mut claim_p = root_p;
  let mut claim_q = root_q;

  for k in 0..d {
    let lambda = transcript.squeeze(b"gkr_lambda")?;
    let half = 1usize << k; // size of the x-cube for this layer's sumcheck

    let mut a0 = levels_p[k + 1][0..half].to_vec();
    let mut a1 = levels_p[k + 1][half..2 * half].to_vec();
    let mut b0 = levels_q[k + 1][0..half].to_vec();
    let mut b1 = levels_q[k + 1][half..2 * half].to_vec();

    // Gruen eq-factoring: the integrand is eq(ρ; x)·g(x) with
    // g = p0·q1 + p1·q0 + λ·q0·q1 of degree 2 per variable, and
    // eq(ρ; (r_{<j}, t, x_rest)) = E_pref · E_j(t) · eq(ρ_{>j}, x_rest)
    // where E_j(t) = (1−t)(1−ρ_j) + t·ρ_j is linear. So the round
    // polynomial is s(t) = E_pref·E_j(t)·h(t) with h of degree 2: per
    // index we only accumulate h(0) and the quadratic coefficient h(∞)
    // (from table diffs), and recover h(1) from the running claim via
    // s(0) + s(1) = claim. The eq table is never extrapolated or bound;
    // the per-round weights are the suffix tables eq(ρ_{>j}, ·),
    // precomputed here in one backward pass (suffix[j] covers ρ_{j..};
    // round j uses suffix[j+1]). The emitted s(0..3) are exactly the
    // same field elements as the unfactored evaluation, so the
    // transcript and verifier are unchanged.
    let mut suffix: Vec<Vec<E::Scalar>> = vec![Vec::new(); k + 1];
    suffix[k] = vec![E::Scalar::ONE];
    for j in (1..k).rev() {
      let prev = &suffix[j + 1];
      let c = point[j];
      let mut tbl = vec![E::Scalar::ZERO; prev.len() * 2];
      let (lo, hi) = tbl.split_at_mut(prev.len());
      let one_minus_c = E::Scalar::ONE - c;
      for (i, &p) in prev.iter().enumerate() {
        lo[i] = one_minus_c * p;
        hi[i] = c * p;
      }
      suffix[j] = tbl;
    }

    let mut e_pref = E::Scalar::ONE;
    let mut claim = claim_p + lambda * claim_q;
    let mut round_polys = Vec::with_capacity(k);
    let mut challenges = Vec::with_capacity(k);

    for round in 0..k {
      let h = a0.len() / 2;
      let w = &suffix[round + 1];
      debug_assert_eq!(w.len(), h);
      let c = point[round];

      // Accumulate h(0) and h(∞) (the quadratic coefficient of h).
      let eval_at = |i: usize| -> [E::Scalar; 2] {
        let a0d = a0[i + h] - a0[i];
        let a1d = a1[i + h] - a1[i];
        let b0d = b0[i + h] - b0[i];
        let b1d = b1[i + h] - b1[i];
        let g0 = a0[i] * b1[i] + a1[i] * b0[i] + lambda * (b0[i] * b1[i]);
        let ginf = a0d * b1d + a1d * b0d + lambda * (b0d * b1d);
        [w[i] * g0, w[i] * ginf]
      };
      let add2 = |mut a: [E::Scalar; 2], b: [E::Scalar; 2]| {
        a[0] += b[0];
        a[1] += b[1];
        a
      };
      let [h0, hinf] = if h >= PAR_THRESHOLD {
        (0..h)
          .into_par_iter()
          .fold(|| [E::Scalar::ZERO; 2], |acc, i| add2(acc, eval_at(i)))
          .reduce(|| [E::Scalar::ZERO; 2], add2)
      } else {
        (0..h).fold([E::Scalar::ZERO; 2], |acc, i| add2(acc, eval_at(i)))
      };

      // s(t) = E_pref·E(t)·h(t), E(t) = (1−t)(1−c) + t·c.
      let e0 = e_pref * (E::Scalar::ONE - c);
      let e1 = e_pref * c;
      let s0 = e0 * h0;
      let s1 = claim - s0;
      // h(1) from the claim; direct evaluation as a (negligible-
      // probability) fallback when E_pref·c isn't invertible.
      let h1 = match Option::<E::Scalar>::from(e1.invert()) {
        Some(e1_inv) => s1 * e1_inv,
        None => {
          let acc = |i: usize| {
            w[i]
              * (a0[i + h] * b1[i + h] + a1[i + h] * b0[i + h] + lambda * (b0[i + h] * b1[i + h]))
          };
          (0..h).fold(E::Scalar::ZERO, |s, i| s + acc(i))
        }
      };
      // h(t) = h0 + b·t + c2·t².
      let c2 = hinf;
      let bq = h1 - h0 - c2;
      let two = E::Scalar::from(2);
      let three = E::Scalar::from(3);
      let h2 = h0 + two * bq + E::Scalar::from(4) * c2;
      let h3 = h0 + three * bq + E::Scalar::from(9) * c2;
      let e2 = e_pref * (three * c - E::Scalar::ONE);
      let e3 = e_pref * (E::Scalar::from(5) * c - two);
      let s = [s0, s1, e2 * h2, e3 * h3];

      for v in &s {
        transcript.absorb(b"gkr_rp", v);
      }
      let ri = transcript.squeeze(b"gkr_chal")?;
      bind_top(&mut a0, ri);
      bind_top(&mut a1, ri);
      bind_top(&mut b0, ri);
      bind_top(&mut b1, ri);
      claim = eval_cubic::<E::Scalar>(&s, ri);
      e_pref *= (E::Scalar::ONE - c) * (E::Scalar::ONE - ri) + c * ri;
      round_polys.push(s);
      challenges.push(ri);
    }

    let (p0, p1, q0, q1) = (a0[0], a1[0], b0[0], b1[0]);
    transcript.absorb(b"gkr_p0", &p0);
    transcript.absorb(b"gkr_p1", &p1);
    transcript.absorb(b"gkr_q0", &q0);
    transcript.absorb(b"gkr_q1", &q1);
    let c = transcript.squeeze(b"gkr_c")?;

    // Next layer's claim point is (c, challenges) with c the top variable.
    let mut next_point = Vec::with_capacity(k + 1);
    next_point.push(c);
    next_point.extend_from_slice(&challenges);
    point = next_point;
    claim_p = (E::Scalar::ONE - c) * p0 + c * p1;
    claim_q = (E::Scalar::ONE - c) * q0 + c * q1;

    layers.push(GkrLayerProof {
      round_polys,
      p0,
      p1,
      q0,
      q1,
    });
  }

  Ok(GkrOut {
    proof: GkrProof { layers },
    root_p,
    root_q,
    leaf_point: point,
    leaf_p: claim_p,
    leaf_q: claim_q,
  })
}

/// Verify a fractional-sum GKR proof against the claimed root fraction and the
/// expected number of layers `d`. Returns the reduced leaf claim
/// `(point, p_leaf(point), q_leaf(point))`.
fn gkr_verify<E: Engine>(
  root_p: E::Scalar,
  root_q: E::Scalar,
  d: usize,
  proof: &GkrProof<E>,
  transcript: &mut E::TE,
) -> Result<(Vec<E::Scalar>, E::Scalar, E::Scalar), SpartanError> {
  if proof.layers.len() != d {
    return Err(SpartanError::ProofVerifyError {
      reason: "logup-gkr: wrong number of layers".to_string(),
    });
  }
  transcript.absorb(b"gkr_root_p", &root_p);
  transcript.absorb(b"gkr_root_q", &root_q);

  let mut point: Vec<E::Scalar> = Vec::new();
  let mut claim_p = root_p;
  let mut claim_q = root_q;

  for k in 0..d {
    let lp = &proof.layers[k];
    if lp.round_polys.len() != k {
      return Err(SpartanError::ProofVerifyError {
        reason: "logup-gkr: wrong round count in layer".to_string(),
      });
    }
    let lambda = transcript.squeeze(b"gkr_lambda")?;
    let mut claim = claim_p + lambda * claim_q;
    let mut challenges = Vec::with_capacity(k);

    for s in &lp.round_polys {
      // Sumcheck consistency: s(0) + s(1) == running claim.
      if s[0] + s[1] != claim {
        return Err(SpartanError::ProofVerifyError {
          reason: "logup-gkr: sumcheck round mismatch".to_string(),
        });
      }
      for v in s {
        transcript.absorb(b"gkr_rp", v);
      }
      let ri = transcript.squeeze(b"gkr_chal")?;
      claim = eval_cubic::<E::Scalar>(s, ri);
      challenges.push(ri);
    }

    transcript.absorb(b"gkr_p0", &lp.p0);
    transcript.absorb(b"gkr_p1", &lp.p1);
    transcript.absorb(b"gkr_q0", &lp.q0);
    transcript.absorb(b"gkr_q1", &lp.q1);

    // Final sumcheck check against the gate relation:
    //   eq(point, ρ')·[p0·q1 + p1·q0 + λ·q0·q1] == claim.
    let eq_val = EqPolynomial::new(point.clone()).evaluate(&challenges);
    let gate = lp.p0 * lp.q1 + lp.p1 * lp.q0 + lambda * (lp.q0 * lp.q1);
    if eq_val * gate != claim {
      return Err(SpartanError::ProofVerifyError {
        reason: "logup-gkr: layer gate check failed".to_string(),
      });
    }

    let c = transcript.squeeze(b"gkr_c")?;
    let mut next_point = Vec::with_capacity(k + 1);
    next_point.push(c);
    next_point.extend_from_slice(&challenges);
    point = next_point;
    claim_p = (E::Scalar::ONE - c) * lp.p0 + c * lp.p1;
    claim_q = (E::Scalar::ONE - c) * lp.q0 + c * lp.q1;
  }

  Ok((point, claim_p, claim_q))
}

/// Evaluation claims a [`LogUpRangeProof`] reduces to, for the caller to
/// discharge with PCS openings.
#[derive(Clone, Debug)]
pub struct RangeClaims<E: Engine> {
  /// The LogUp challenge `r`.
  pub r: E::Scalar,
  /// Point at which the witness MLE must be opened (`ρ_L`).
  pub wit_point: Vec<E::Scalar>,
  /// Claimed `w(ρ_L)` — open the witness commitment here and check equality.
  pub wit_eval: E::Scalar,
  /// Point at which the multiplicity MLE must be opened (`ρ_R`).
  pub mult_point: Vec<E::Scalar>,
  /// Claimed `m(ρ_R)` — open the multiplicity commitment here and check.
  pub mult_eval: E::Scalar,
}

/// A LogUp-GKR proof that a witness vector is range-bounded by `[0, 2^bits)`.
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(bound = "")]
pub struct LogUpRangeProof<E: Engine> {
  p_lhs_root: E::Scalar,
  q_lhs_root: E::Scalar,
  p_rhs_root: E::Scalar,
  q_rhs_root: E::Scalar,
  lhs_gkr: GkrProof<E>,
  rhs_gkr: GkrProof<E>,
}

impl<E: Engine> LogUpRangeProof<E> {
  /// Number of variables of the padded witness polynomial (`log2` of the
  /// witness leaf count). The witness commitment must be over this many
  /// variables, with the trailing slots padded with the value `0`.
  pub fn witness_num_vars(witness_len: usize) -> usize {
    witness_len.max(1).next_power_of_two().trailing_zeros() as usize
  }

  /// The multiplicity table `m_j = #{i : w[i] = j}` over `[0, 2^bits)`,
  /// including the power-of-two padding `0`s that [`Self::prove`] appends.
  /// A caller that commits the multiplicity polynomial (which it must do
  /// — and absorb — *before* the transcript point where `prove` squeezes
  /// the LogUp challenge `r`) gets the exact vector `prove` will use.
  /// Errors if any witness value is out of range.
  pub fn multiplicities(bits: usize, witness: &[u64]) -> Result<Vec<u64>, SpartanError> {
    let table = 1usize << bits;
    let n = witness.len().max(1).next_power_of_two();
    let mut mult = vec![0u64; table];
    for &w in witness {
      let idx = w as usize;
      if idx >= table {
        return Err(SpartanError::InvalidInputLength {
          reason: format!("logup-gkr: witness value {w} >= 2^{bits}"),
        });
      }
      mult[idx] += 1;
    }
    mult[0] += (n - witness.len()) as u64;
    Ok(mult)
  }

  /// Prove that every value in `witness` lies in `[0, 2^bits)`.
  ///
  /// Returns the proof and the [`RangeClaims`] (the witness and multiplicity
  /// evaluation claims to discharge via PCS). The witness is conceptually
  /// padded to the next power of two with the value `0` (the multiplicity of
  /// `0` is bumped accordingly), so the committed witness polynomial has
  /// [`Self::witness_num_vars`] variables.
  pub fn prove(
    bits: usize,
    witness: &[u64],
    transcript: &mut E::TE,
  ) -> Result<(Self, RangeClaims<E>), SpartanError> {
    if witness.is_empty() {
      return Err(SpartanError::InvalidInputLength {
        reason: "logup-gkr: empty witness".to_string(),
      });
    }
    let table = 1usize << bits;
    let n = witness.len().next_power_of_two();

    // Multiplicities over the table, plus the padding `0`s. Callers that
    // commit the multiplicity polynomial (they must, before this point in
    // the transcript) obtain the identical vector from
    // [`Self::multiplicities`].
    let mult = Self::multiplicities(bits, witness)?;

    transcript.dom_sep(b"logup_range");
    let r = transcript.squeeze(b"logup_r")?;

    // Witness-side leaves: (1, r + w[i]); padding leaves use w = 0.
    let p_lhs = vec![E::Scalar::ONE; n];
    let mut q_lhs = vec![E::Scalar::ZERO; n];
    for (i, slot) in q_lhs.iter_mut().enumerate() {
      let w = if i < witness.len() { witness[i] } else { 0 };
      *slot = r + E::Scalar::from(w);
    }

    // Table-side leaves: (m_j, r + j).
    let mut p_rhs = vec![E::Scalar::ZERO; table];
    let mut q_rhs = vec![E::Scalar::ZERO; table];
    for j in 0..table {
      p_rhs[j] = E::Scalar::from(mult[j]);
      q_rhs[j] = r + E::Scalar::from(j as u64);
    }

    let lhs = gkr_prove::<E>(p_lhs, q_lhs, transcript)?;
    let rhs = gkr_prove::<E>(p_rhs, q_rhs, transcript)?;

    let claims = RangeClaims {
      r,
      wit_point: lhs.leaf_point.clone(),
      wit_eval: lhs.leaf_q - r, // q_leaf = r + w  ⇒  w = q_leaf - r
      mult_point: rhs.leaf_point.clone(),
      mult_eval: rhs.leaf_p, // p_leaf = m
    };

    Ok((
      LogUpRangeProof {
        p_lhs_root: lhs.root_p,
        q_lhs_root: lhs.root_q,
        p_rhs_root: rhs.root_p,
        q_rhs_root: rhs.root_q,
        lhs_gkr: lhs.proof,
        rhs_gkr: rhs.proof,
      },
      claims,
    ))
  }

  /// Verify the proof and return the [`RangeClaims`] the caller must discharge
  /// (witness opening at `wit_point == wit_eval`, multiplicity opening at
  /// `mult_point == mult_eval`).
  pub fn verify(
    &self,
    bits: usize,
    transcript: &mut E::TE,
  ) -> Result<RangeClaims<E>, SpartanError> {
    let d_lhs = self.lhs_gkr.layers.len();

    transcript.dom_sep(b"logup_range");
    let r = transcript.squeeze(b"logup_r")?;

    let (lhs_point, lhs_p, lhs_q) = gkr_verify::<E>(
      self.p_lhs_root,
      self.q_lhs_root,
      d_lhs,
      &self.lhs_gkr,
      transcript,
    )?;
    let (rhs_point, rhs_p, rhs_q) = gkr_verify::<E>(
      self.p_rhs_root,
      self.q_rhs_root,
      bits,
      &self.rhs_gkr,
      transcript,
    )?;

    // LogUp identity: sum_LHS == sum_RHS  ⇔  P_L/Q_L == P_R/Q_R.
    if self.q_lhs_root == E::Scalar::ZERO || self.q_rhs_root == E::Scalar::ZERO {
      return Err(SpartanError::ProofVerifyError {
        reason: "logup-gkr: zero denominator (challenge hit a pole)".to_string(),
      });
    }
    if self.p_lhs_root * self.q_rhs_root != self.p_rhs_root * self.q_lhs_root {
      return Err(SpartanError::ProofVerifyError {
        reason: "logup-gkr: LogUp identity failed (value out of range)".to_string(),
      });
    }

    // Structured leaf checks. Witness numerators are all 1, so the leaf
    // numerator MLE must evaluate to 1.
    if lhs_p != E::Scalar::ONE {
      return Err(SpartanError::ProofVerifyError {
        reason: "logup-gkr: witness numerator leaf != 1".to_string(),
      });
    }
    // Table denominators are r + idx(j); the verifier reconstructs idx in
    // closed form.
    if rhs_q != r + idx_mle_eval::<E::Scalar>(&rhs_point) {
      return Err(SpartanError::ProofVerifyError {
        reason: "logup-gkr: table index leaf mismatch".to_string(),
      });
    }

    Ok(RangeClaims {
      r,
      wit_point: lhs_point,
      wit_eval: lhs_q - r,
      mult_point: rhs_point,
      mult_eval: rhs_p,
    })
  }
}

/// Evaluation claims a [`LogUpMultiRangeProof`] reduces to: one
/// `(point, eval)` pair per witness tree (in input order), plus the single
/// multiplicity claim. Each must be discharged with a PCS opening.
#[derive(Clone, Debug)]
pub struct MultiRangeClaims<E: Engine> {
  /// The (shared) LogUp challenge `r`.
  pub r: E::Scalar,
  /// `(ρ_i, w_i(ρ_i))` per witness tree, in input order.
  pub wit_claims: Vec<(Vec<E::Scalar>, E::Scalar)>,
  /// Point at which the shared multiplicity MLE must be opened.
  pub mult_point: Vec<E::Scalar>,
  /// Claimed `m(mult_point)`.
  pub mult_eval: E::Scalar,
}

/// A LogUp-GKR proof that *several* witness vectors are all range-bounded
/// by `[0, 2^bits)` against ONE shared multiplicity table. The lookup
/// identity is additive, so each witness gets its own fraction tree
/// (reduced to an opening of its own commitment) while the table side —
/// the `2^bits`-leaf tree and the multiplicity commitment — is paid once:
///
/// ```text
///   Σ_b Σ_i 1/(r + w_b[i])  =  Σ_j m_j/(r + j)
/// ```
///
/// The verifier sums the witness root fractions and cross-multiplies
/// against the table root. Every witness vector must already be a power
/// of two long (committed polynomials are); the table counts them all.
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(bound = "")]
pub struct LogUpMultiRangeProof<E: Engine> {
  /// Per-witness-tree root fraction `(P_b, Q_b)`.
  wit_roots: Vec<(E::Scalar, E::Scalar)>,
  p_rhs_root: E::Scalar,
  q_rhs_root: E::Scalar,
  wit_gkrs: Vec<GkrProof<E>>,
  rhs_gkr: GkrProof<E>,
}

impl<E: Engine> LogUpMultiRangeProof<E> {
  /// The shared multiplicity table over all witness vectors:
  /// `m_j = Σ_b #{i : w_b[i] = j}`. Each witness must be power-of-two
  /// long (no implicit padding — committed polys already are). Callers
  /// must commit this table and absorb it *before* the transcript point
  /// where [`Self::prove`] squeezes the LogUp challenge `r`.
  pub fn multiplicities(bits: usize, witnesses: &[&[u64]]) -> Result<Vec<u64>, SpartanError> {
    let table = 1usize << bits;
    let mut mult = vec![0u64; table];
    for (b, witness) in witnesses.iter().enumerate() {
      if witness.is_empty() || !witness.len().is_power_of_two() {
        return Err(SpartanError::InvalidInputLength {
          reason: format!(
            "logup-gkr multi: witness {b} length {} is not a positive power of two",
            witness.len()
          ),
        });
      }
      for &w in witness.iter() {
        let idx = w as usize;
        if idx >= table {
          return Err(SpartanError::InvalidInputLength {
            reason: format!("logup-gkr multi: witness {b} value {w} >= 2^{bits}"),
          });
        }
        mult[idx] += 1;
      }
    }
    Ok(mult)
  }

  /// Prove that every value of every witness lies in `[0, 2^bits)`.
  pub fn prove(
    bits: usize,
    witnesses: &[&[u64]],
    transcript: &mut E::TE,
  ) -> Result<(Self, MultiRangeClaims<E>), SpartanError> {
    if witnesses.is_empty() {
      return Err(SpartanError::InvalidInputLength {
        reason: "logup-gkr multi: no witnesses".to_string(),
      });
    }
    let table = 1usize << bits;
    let mult = Self::multiplicities(bits, witnesses)?;

    transcript.dom_sep(b"logup_multi_range");
    let r = transcript.squeeze(b"logup_r")?;

    // One fraction tree per witness: leaves (1, r + w_b[i]).
    let mut wit_roots = Vec::with_capacity(witnesses.len());
    let mut wit_gkrs = Vec::with_capacity(witnesses.len());
    let mut wit_claims = Vec::with_capacity(witnesses.len());
    for witness in witnesses {
      let p = vec![E::Scalar::ONE; witness.len()];
      let q: Vec<E::Scalar> = witness.iter().map(|&w| r + E::Scalar::from(w)).collect();
      let out = gkr_prove::<E>(p, q, transcript)?;
      wit_roots.push((out.root_p, out.root_q));
      wit_claims.push((out.leaf_point, out.leaf_q - r));
      wit_gkrs.push(out.proof);
    }

    // One shared table tree: leaves (m_j, r + j).
    let p_rhs: Vec<E::Scalar> = mult.iter().map(|&m| E::Scalar::from(m)).collect();
    let q_rhs: Vec<E::Scalar> = (0..table).map(|j| r + E::Scalar::from(j as u64)).collect();
    let rhs = gkr_prove::<E>(p_rhs, q_rhs, transcript)?;

    let claims = MultiRangeClaims {
      r,
      wit_claims,
      mult_point: rhs.leaf_point,
      mult_eval: rhs.leaf_p,
    };

    Ok((
      LogUpMultiRangeProof {
        wit_roots,
        p_rhs_root: rhs.root_p,
        q_rhs_root: rhs.root_q,
        wit_gkrs,
        rhs_gkr: rhs.proof,
      },
      claims,
    ))
  }

  /// Verify against the expected per-witness tree depths (`log2` of each
  /// committed witness polynomial's length — the caller MUST pin these,
  /// otherwise a prover could range-check smaller polynomials). Returns
  /// the [`MultiRangeClaims`] to discharge via PCS openings.
  pub fn verify(
    &self,
    bits: usize,
    expected_wit_depths: &[usize],
    transcript: &mut E::TE,
  ) -> Result<MultiRangeClaims<E>, SpartanError> {
    if self.wit_gkrs.len() != expected_wit_depths.len()
      || self.wit_roots.len() != expected_wit_depths.len()
      || expected_wit_depths.is_empty()
    {
      return Err(SpartanError::ProofVerifyError {
        reason: "logup-gkr multi: witness tree count mismatch".to_string(),
      });
    }

    transcript.dom_sep(b"logup_multi_range");
    let r = transcript.squeeze(b"logup_r")?;

    let mut wit_claims = Vec::with_capacity(self.wit_gkrs.len());
    for (i, gkr) in self.wit_gkrs.iter().enumerate() {
      let (root_p, root_q) = self.wit_roots[i];
      let (point, leaf_p, leaf_q) =
        gkr_verify::<E>(root_p, root_q, expected_wit_depths[i], gkr, transcript)?;
      // Witness numerators are all 1.
      if leaf_p != E::Scalar::ONE {
        return Err(SpartanError::ProofVerifyError {
          reason: format!("logup-gkr multi: witness {i} numerator leaf != 1"),
        });
      }
      wit_claims.push((point, leaf_q - r));
    }

    let (rhs_point, rhs_p, rhs_q) = gkr_verify::<E>(
      self.p_rhs_root,
      self.q_rhs_root,
      bits,
      &self.rhs_gkr,
      transcript,
    )?;
    // Table denominators are r + idx(j), checked in closed form.
    if rhs_q != r + idx_mle_eval::<E::Scalar>(&rhs_point) {
      return Err(SpartanError::ProofVerifyError {
        reason: "logup-gkr multi: table index leaf mismatch".to_string(),
      });
    }

    // Summed LogUp identity: Σ_b P_b/Q_b == P_R/Q_R, via fraction folding
    // and one cross-product. Reject zero denominators (pole hits).
    if self.q_rhs_root == E::Scalar::ZERO {
      return Err(SpartanError::ProofVerifyError {
        reason: "logup-gkr multi: zero table denominator".to_string(),
      });
    }
    let mut sum_p = E::Scalar::ZERO;
    let mut sum_q = E::Scalar::ONE;
    for (i, &(p_b, q_b)) in self.wit_roots.iter().enumerate() {
      if q_b == E::Scalar::ZERO {
        return Err(SpartanError::ProofVerifyError {
          reason: format!("logup-gkr multi: zero denominator in witness {i}"),
        });
      }
      sum_p = sum_p * q_b + p_b * sum_q;
      sum_q *= q_b;
    }
    if sum_p * self.q_rhs_root != self.p_rhs_root * sum_q {
      return Err(SpartanError::ProofVerifyError {
        reason: "logup-gkr multi: LogUp identity failed (value out of range)".to_string(),
      });
    }

    Ok(MultiRangeClaims {
      r,
      wit_claims,
      mult_point: rhs_point,
      mult_eval: rhs_p,
    })
  }
}

#[cfg(test)]
mod tests {
  use super::*;
  use crate::provider::{PallasHyraxEngine, T256HyraxEngine};
  use ff::Field;
  use rand::{Rng, SeedableRng, rngs::StdRng};

  /// Direct MLE evaluation of a dense table at `point` (point[0] = MSB),
  /// matching the GKR's big-endian variable order.
  fn mle_eval<F: PrimeField>(table: &[F], point: &[F]) -> F {
    let chis = EqPolynomial::evals_from_points(point);
    table.iter().zip(chis.iter()).map(|(z, c)| *z * *c).sum()
  }

  #[test]
  fn idx_mle_matches_dense() {
    type F = <PallasHyraxEngine as Engine>::Scalar;
    for bits in 1..=6usize {
      let table: Vec<F> = (0..(1u64 << bits)).map(F::from).collect();
      let point: Vec<F> = (0..bits)
        .map(|i| F::from((7 * i as u64 + 3) % 101))
        .collect();
      assert_eq!(idx_mle_eval::<F>(&point), mle_eval(&table, &point));
    }
  }

  /// End-to-end: an in-range witness verifies, and the reduced witness /
  /// multiplicity claims match the true MLE evaluations.
  fn range_roundtrip_with<E: Engine>(bits: usize, witness: &[u64])
  where
    <E::Scalar as crate::traits::mod_engine::SumcheckField>::Params: Default,
  {
    type TEof<E> = <E as Engine>::TE;
    let mut tp = <TEof<E>>::new(b"logup_test");
    let (proof, claims_p) = LogUpRangeProof::<E>::prove(bits, witness, &mut tp).unwrap();

    let mut tv = <TEof<E>>::new(b"logup_test");
    let claims_v = proof.verify(bits, &mut tv).unwrap();

    // Prover and verifier agree on the reduced claims.
    assert_eq!(claims_p.r, claims_v.r);
    assert_eq!(claims_p.wit_point, claims_v.wit_point);
    assert_eq!(claims_p.wit_eval, claims_v.wit_eval);
    assert_eq!(claims_p.mult_point, claims_v.mult_point);
    assert_eq!(claims_p.mult_eval, claims_v.mult_eval);

    // The reduced claims match the true MLEs of the padded witness and the
    // multiplicity table (this is what a PCS opening would confirm).
    let table = 1usize << bits;
    let n = witness.len().next_power_of_two();
    let mut w_tbl = vec![E::Scalar::ZERO; n];
    for (i, slot) in w_tbl.iter_mut().enumerate() {
      let w = if i < witness.len() { witness[i] } else { 0 };
      *slot = E::Scalar::from(w);
    }
    let mut mult = vec![0u64; table];
    for &w in witness {
      mult[w as usize] += 1;
    }
    mult[0] += (n - witness.len()) as u64;
    let m_tbl: Vec<E::Scalar> = mult.iter().map(|&m| E::Scalar::from(m)).collect();

    assert_eq!(claims_v.wit_eval, mle_eval(&w_tbl, &claims_v.wit_point));
    assert_eq!(claims_v.mult_eval, mle_eval(&m_tbl, &claims_v.mult_point));
  }

  #[test]
  fn range_roundtrips_small() {
    range_roundtrip_with::<PallasHyraxEngine>(4, &[3, 7, 3, 0, 15, 1, 9, 3]);
  }

  #[test]
  fn range_roundtrips_non_power_of_two_witness() {
    // 5 witnesses → padded to 8 with value 0.
    range_roundtrip_with::<PallasHyraxEngine>(4, &[1, 2, 14, 14, 0]);
  }

  #[test]
  fn range_roundtrips_t256_bits8() {
    let mut rng = StdRng::seed_from_u64(42);
    let witness: Vec<u64> = (0..200).map(|_| rng.gen_range(0..256)).collect();
    range_roundtrip_with::<T256HyraxEngine>(8, &witness);
  }

  #[test]
  fn prove_rejects_out_of_range() {
    type E = PallasHyraxEngine;
    let mut tp = <E as Engine>::TE::new(b"logup_test");
    // 16 >= 2^4 = 16, out of range.
    assert!(LogUpRangeProof::<E>::prove(4, &[1, 2, 16], &mut tp).is_err());
  }

  #[test]
  fn verify_rejects_tampered_root() {
    type E = PallasHyraxEngine;
    let mut tp = <E as Engine>::TE::new(b"logup_test");
    let (mut proof, _) = LogUpRangeProof::<E>::prove(4, &[3, 7, 3, 0], &mut tp).unwrap();
    // Corrupt the witness-side numerator root: breaks the LogUp identity.
    proof.p_lhs_root += <E as Engine>::Scalar::ONE;
    let mut tv = <E as Engine>::TE::new(b"logup_test");
    assert!(proof.verify(4, &mut tv).is_err());
  }

  #[test]
  fn verify_rejects_tampered_layer() {
    type E = PallasHyraxEngine;
    let mut tp = <E as Engine>::TE::new(b"logup_test");
    let (mut proof, _) =
      LogUpRangeProof::<E>::prove(4, &[3, 7, 3, 0, 1, 2, 5, 5], &mut tp).unwrap();
    // Corrupt a sumcheck round polynomial deep in the table tree.
    let last = proof.rhs_gkr.layers.len() - 1;
    proof.rhs_gkr.layers[last].round_polys[0][2] += <E as Engine>::Scalar::ONE;
    let mut tv = <E as Engine>::TE::new(b"logup_test");
    assert!(proof.verify(4, &mut tv).is_err());
  }

  /// Multi-witness roundtrip: several trees of different sizes against one
  /// shared table; reduced claims match the true MLEs.
  #[test]
  fn multi_range_roundtrips() {
    type E = PallasHyraxEngine;
    type F = <E as Engine>::Scalar;
    let w0: Vec<u64> = vec![3, 7, 3, 0, 15, 1, 9, 3]; // len 8
    let w1: Vec<u64> = vec![14, 0]; // len 2
    let w2: Vec<u64> = vec![5, 5, 5, 5, 2, 2, 1, 0, 0, 0, 0, 0, 0, 0, 0, 7]; // len 16
    let witnesses: Vec<&[u64]> = vec![&w0, &w1, &w2];

    let mut tp = <E as Engine>::TE::new(b"logup_multi");
    let (proof, claims_p) = LogUpMultiRangeProof::<E>::prove(4, &witnesses, &mut tp).unwrap();

    let depths = [3usize, 1, 4];
    let mut tv = <E as Engine>::TE::new(b"logup_multi");
    let claims_v = proof.verify(4, &depths, &mut tv).unwrap();

    assert_eq!(claims_p.r, claims_v.r);
    assert_eq!(claims_p.wit_claims.len(), 3);

    // Every reduced witness claim matches the true MLE of its vector, and
    // the multiplicity claim matches the shared table's MLE.
    for (b, witness) in witnesses.iter().enumerate() {
      let tbl: Vec<F> = witness.iter().map(|&w| F::from(w)).collect();
      let (point, eval) = &claims_v.wit_claims[b];
      assert_eq!(*eval, mle_eval(&tbl, point), "witness {b} claim mismatch");
    }
    let mult = LogUpMultiRangeProof::<E>::multiplicities(4, &witnesses).unwrap();
    let m_tbl: Vec<F> = mult.iter().map(|&m| F::from(m)).collect();
    assert_eq!(claims_v.mult_eval, mle_eval(&m_tbl, &claims_v.mult_point));
  }

  /// Multi-witness: out-of-range value rejected at prove; tampered roots
  /// and wrong expected depths rejected at verify.
  #[test]
  fn multi_range_rejects_bad() {
    type E = PallasHyraxEngine;
    let w0: Vec<u64> = vec![3, 16, 0, 1]; // 16 out of range for bits=4
    assert!(
      LogUpMultiRangeProof::<E>::prove(4, &[&w0], &mut <E as Engine>::TE::new(b"m")).is_err()
    );

    let w0: Vec<u64> = vec![3, 7, 3, 0];
    let w1: Vec<u64> = vec![1, 2];
    let mut tp = <E as Engine>::TE::new(b"logup_multi");
    let (proof, _) = LogUpMultiRangeProof::<E>::prove(4, &[&w0, &w1], &mut tp).unwrap();

    // Tampered witness root breaks the summed identity.
    let mut bad = proof.clone();
    bad.wit_roots[1].0 += <E as Engine>::Scalar::ONE;
    let mut tv = <E as Engine>::TE::new(b"logup_multi");
    assert!(bad.verify(4, &[2, 1], &mut tv).is_err());

    // Wrong pinned depth rejected.
    let mut tv = <E as Engine>::TE::new(b"logup_multi");
    assert!(proof.verify(4, &[2, 2], &mut tv).is_err());

    // Honest proof still verifies.
    let mut tv = <E as Engine>::TE::new(b"logup_multi");
    assert!(proof.verify(4, &[2, 1], &mut tv).is_ok());
  }
}
