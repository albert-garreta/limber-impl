//! Phase-2 Integer Mod-R1CS SNARK driver, generic over `M: ModEngine`.
//!
//! Mirrors `crate::imod_spartan` but the sumcheck arithmetic runs over
//! `M::Scalar` (a `SumcheckField` — e.g. `DynPrime<4>`), and the PCS is
//! `M::ModPCS`. The `p = q` shortcut (and the absence of limb-splitting)
//! lives inside the Mod-PCS impl, not here: this driver is the same Spartan
//! protocol with the dynamic-prime substitutions applied.
//!
//! Single witness segment; no shared/precommitted/rest split; no limb
//! decomposition; no range checks; no BDDT first-round optimization.

use crate::{
  errors::SpartanError,
  imod_r1cs_modp::{IntModR1CSInstanceModp, IntModR1CSShapeModp, IntModR1CSWitnessModp},
  math::Math,
  polys_modp::{eq::EqPolynomial, multilinear::MultilinearPolynomial},
  start_span,
  sumcheck_modp::SumcheckProof,
  traits::{
    mod_engine::{ModEngine, ModPCSEngineTrait, SumcheckEngine, SumcheckField},
    transcript::{ByteTranscript, TranscriptEngineTrait},
  },
};
use rayon::prelude::*;
use tracing::info;

type MScalar<M> = <M as SumcheckEngine>::Scalar;
type MParams<M> = <MScalar<M> as SumcheckField>::Params;
type ModPCS<M> = <M as ModEngine>::ModPCS;
type ModCK<M> = <ModPCS<M> as ModPCSEngineTrait<M>>::CommitmentKey;
type ModVK<M> = <ModPCS<M> as ModPCSEngineTrait<M>>::VerifierKey;
type ModBlind<M> = <ModPCS<M> as ModPCSEngineTrait<M>>::Blind;
type ModEvalArg<M> = <ModPCS<M> as ModPCSEngineTrait<M>>::EvaluationArgument;

/// Prover key.
#[derive(Clone)]
pub struct IntModSpartanModpProverKey<M: ModEngine> {
  pub(crate) ck: ModCK<M>,
  pub(crate) ck_s: ModCK<M>,
  pub(crate) shape: IntModR1CSShapeModp<M>,
  pub(crate) vk_digest: [u8; 32],
}

impl<M: ModEngine> IntModSpartanModpProverKey<M> {
  /// Commitment key used for `w` and `q`. External callers need this to
  /// build witness/instance pairs via `IntModR1CSWitnessModp::new`.
  pub fn ck(&self) -> &ModCK<M> {
    &self.ck
  }
}

/// Verifier key.
#[derive(Clone)]
pub struct IntModSpartanModpVerifierKey<M: ModEngine> {
  pub(crate) vk_ee: ModVK<M>,
  pub(crate) ck_s: ModCK<M>,
  pub(crate) shape: IntModR1CSShapeModp<M>,
  pub(crate) digest: [u8; 32],
}

impl<M: ModEngine> IntModSpartanModpVerifierKey<M> {
  /// 32-byte verifier-key digest (Keccak256 of the shape).
  pub fn digest(&self) -> [u8; 32] {
    self.digest
  }
}

/// Phase-2 IntMod-R1CS SNARK proof. Serialization is deferred — the
/// dynamic-prime types (`DynPrime`, `MParams`) aren't `Serialize` yet.
#[derive(Clone, Debug)]
pub struct IntModSpartanModpSNARK<M: ModEngine> {
  // outer sumcheck
  sc_outer: SumcheckProof<M>,
  v_a: MScalar<M>,
  v_b: MScalar<M>,
  v_c: MScalar<M>,
  v_m: MScalar<M>,
  v_q: MScalar<M>,
  // inner sumcheck (for w)
  sc_inner: SumcheckProof<M>,
  eval_w: MScalar<M>,
  blind_eval_w: ModBlind<M>,
  eval_arg_w: ModEvalArg<M>,
  // Q opening at r_x
  blind_eval_q: ModBlind<M>,
  eval_arg_q: ModEvalArg<M>,
}

impl<M: ModEngine> IntModSpartanModpSNARK<M> {
  /// Setup: derive prover and verifier keys from the shape.
  pub fn setup(
    shape: IntModR1CSShapeModp<M>,
  ) -> Result<
    (
      IntModSpartanModpProverKey<M>,
      IntModSpartanModpVerifierKey<M>,
    ),
    SpartanError,
  > {
    let (ck, vk_ee) = shape.commitment_key();
    <ModPCS<M> as ModPCSEngineTrait<M>>::precompute_ck(&ck);
    let (ck_s, _) = <ModPCS<M> as ModPCSEngineTrait<M>>::setup(b"ck_s_imod_modp", 1, 1);
    <ModPCS<M> as ModPCSEngineTrait<M>>::precompute_ck(&ck_s);

    let digest = shape.digest();
    let vk = IntModSpartanModpVerifierKey {
      vk_ee,
      ck_s: ck_s.clone(),
      shape: shape.clone(),
      digest,
    };
    let pk = IntModSpartanModpProverKey {
      ck,
      ck_s,
      shape,
      vk_digest: digest,
    };
    Ok((pk, vk))
  }

  /// Prove satisfaction of the IntMod-R1CS instance.
  pub fn prove(
    pk: &IntModSpartanModpProverKey<M>,
    U: &IntModR1CSInstanceModp<M>,
    W: &IntModR1CSWitnessModp<M>,
  ) -> Result<Self, SpartanError> {
    let (_prove_span, prove_t) = start_span!("imod_spartan_modp_prove");
    let params = pk.shape.params.clone();
    let mut transcript = <M::TE as TranscriptEngineTrait<M>>::new_with_params(
      b"IntModSpartanModpSNARK",
      params.clone(),
    );

    transcript.absorb_bytes(b"vk", &pk.vk_digest);
    transcript.absorb(b"comm_w", &U.comm_w);
    transcript.absorb(b"comm_q", &U.comm_q);
    transcript.absorb(b"x", &U.x.as_slice());

    let shape = &pk.shape;
    let num_vars = shape.num_vars;
    let num_cons = shape.num_cons;
    let num_rounds_x = num_cons.log_2();
    let num_rounds_y = num_vars.log_2() + 1;

    let zero = <MScalar<M> as SumcheckField>::zero(&params);
    let one = <MScalar<M> as SumcheckField>::one(&params);

    // z = (W, 1, X), padded to 2*num_vars for the MLE.
    let mut z = Vec::with_capacity(2 * num_vars);
    z.extend_from_slice(&W.w);
    z.push(one);
    z.extend_from_slice(&U.x);
    z.resize(2 * num_vars, zero);

    let z_for_spmv = &z[..num_vars + 1 + shape.num_io];
    let (az, bz, cz) = shape.multiply_vec(z_for_spmv)?;
    let m: Vec<MScalar<M>> = shape.mods.clone();
    let q_for_open = W.q.clone();

    // Outer sumcheck: sum_i eq(i, tau) · (Az·Bz − Cz − M·Q) = 0.
    let tau: Vec<MScalar<M>> = (0..num_rounds_x)
      .map(|_| transcript.squeeze(b"tau"))
      .collect::<Result<Vec<_>, SpartanError>>()?;

    let mut poly_az = MultilinearPolynomial::new(az, params.clone());
    let mut poly_bz = MultilinearPolynomial::new(bz, params.clone());
    let mut poly_cz = MultilinearPolynomial::new(cz, params.clone());
    let mut poly_m = MultilinearPolynomial::new(m, params.clone());
    let mut poly_q = MultilinearPolynomial::new(W.q.clone(), params.clone());

    let (_so_span, so_t) = start_span!("imod_modp_outer_sumcheck");
    let (sc_outer, r_x, outer_claims) = SumcheckProof::<M>::prove_cubic_with_five_inputs(
      &zero,
      tau,
      &mut poly_az,
      &mut poly_bz,
      &mut poly_cz,
      &mut poly_m,
      &mut poly_q,
      &mut transcript,
    )?;
    info!(elapsed_ms = %so_t.elapsed().as_millis(), "imod_modp_outer_sumcheck");

    let v_a = outer_claims[0];
    let v_b = outer_claims[1];
    let v_c = outer_claims[2];
    let v_m = outer_claims[3];
    let v_q = outer_claims[4];

    transcript.absorb(b"outer_claims", &[v_a, v_b, v_c, v_m, v_q].as_slice());

    // Inner sumcheck: sum_y (A(r_x,y) + r·B(r_x,y) + r²·C(r_x,y)) · z(y).
    let r = transcript.squeeze(b"r")?;
    let claim_inner = v_a + r * v_b + r * r * v_c;

    let eq_rx = EqPolynomial::<MScalar<M>>::evals_from_points(&r_x, &params);
    let abc = bind_abc::<M>(shape, &eq_rx, &r);

    debug_assert_eq!(abc.len(), 2 * num_vars);
    debug_assert_eq!(z.len(), 2 * num_vars);

    let mut poly_abc = MultilinearPolynomial::new(abc, params.clone());
    let mut poly_z = MultilinearPolynomial::new(z, params.clone());

    let (_si_span, si_t) = start_span!("imod_modp_inner_sumcheck");
    let (sc_inner, r_y, _claims_inner) = SumcheckProof::<M>::prove_quad(
      &claim_inner,
      num_rounds_y,
      &mut poly_abc,
      &mut poly_z,
      &mut transcript,
    )?;
    info!(elapsed_ms = %si_t.elapsed().as_millis(), "imod_modp_inner_sumcheck");

    // Recover eval_W from eval_Z via Z = (W, 1, X, …):
    //   Z(r_y) = (1 - r_y[0]) · W(r_y[1..]) + r_y[0] · pub(r_y[1..]).
    let eval_z = poly_z[0];
    let eval_x = eval_public_at::<M>(num_rounds_y - 1, &U.x, &r_y[1..], &params);
    let one_minus_r0 = one - r_y[0];
    let inv = one_minus_r0.invert().ok_or(SpartanError::DivisionByZero)?;
    let eval_w = (eval_z - r_y[0] * eval_x) * inv;

    // Mod-PCS open W at r_y[1..].
    let blind_eval_w = <ModPCS<M> as ModPCSEngineTrait<M>>::blind(&pk.ck_s, 1);
    let comm_eval_w =
      <ModPCS<M> as ModPCSEngineTrait<M>>::commit(&pk.ck_s, &[eval_w], &blind_eval_w, false)?;
    let eval_arg_w = <ModPCS<M> as ModPCSEngineTrait<M>>::prove(
      &pk.ck,
      &pk.ck_s,
      &mut transcript,
      &U.comm_w,
      &W.w,
      &W.r_w,
      &r_y[1..],
      &comm_eval_w,
      &blind_eval_w,
    )?;

    // Mod-PCS open Q at r_x.
    let blind_eval_q = <ModPCS<M> as ModPCSEngineTrait<M>>::blind(&pk.ck_s, 1);
    let comm_eval_q =
      <ModPCS<M> as ModPCSEngineTrait<M>>::commit(&pk.ck_s, &[v_q], &blind_eval_q, false)?;
    let eval_arg_q = <ModPCS<M> as ModPCSEngineTrait<M>>::prove(
      &pk.ck,
      &pk.ck_s,
      &mut transcript,
      &U.comm_q,
      &q_for_open,
      &W.r_q,
      &r_x,
      &comm_eval_q,
      &blind_eval_q,
    )?;

    info!(elapsed_ms = %prove_t.elapsed().as_millis(), "imod_spartan_modp_prove");
    Ok(Self {
      sc_outer,
      v_a,
      v_b,
      v_c,
      v_m,
      v_q,
      sc_inner,
      eval_w,
      blind_eval_w,
      eval_arg_w,
      blind_eval_q,
      eval_arg_q,
    })
  }

  /// Verify the SNARK against an instance.
  pub fn verify(
    &self,
    vk: &IntModSpartanModpVerifierKey<M>,
    U: &IntModR1CSInstanceModp<M>,
  ) -> Result<(), SpartanError> {
    let (_verify_span, verify_t) = start_span!("imod_spartan_modp_verify");
    let params = vk.shape.params.clone();
    let mut transcript = <M::TE as TranscriptEngineTrait<M>>::new_with_params(
      b"IntModSpartanModpSNARK",
      params.clone(),
    );

    transcript.absorb_bytes(b"vk", &vk.digest);
    transcript.absorb(b"comm_w", &U.comm_w);
    transcript.absorb(b"comm_q", &U.comm_q);
    transcript.absorb(b"x", &U.x.as_slice());

    let shape = &vk.shape;
    let num_vars = shape.num_vars;
    let num_cons = shape.num_cons;
    let num_rounds_x = num_cons.log_2();
    let num_rounds_y = num_vars.log_2() + 1;

    let zero = <MScalar<M> as SumcheckField>::zero(&params);
    let one = <MScalar<M> as SumcheckField>::one(&params);

    // Outer SC verification.
    let tau: Vec<MScalar<M>> = (0..num_rounds_x)
      .map(|_| transcript.squeeze(b"tau"))
      .collect::<Result<Vec<_>, SpartanError>>()?;

    let (claim_outer_final, r_x) =
      self
        .sc_outer
        .verify(zero, num_rounds_x, 3, &params, &mut transcript)?;

    // v_m matches the public mods MLE at r_x.
    let v_m_expected = dense_evaluate::<M>(&shape.mods, &r_x, &params);
    if v_m_expected != self.v_m {
      return Err(SpartanError::InvalidSumcheckProof);
    }

    // Reconstruct the outer-SC final claim.
    let eq_tau_rx = EqPolynomial::<MScalar<M>>::new(tau, params.clone()).evaluate(&r_x);
    let outer_final_expected = eq_tau_rx * (self.v_a * self.v_b - self.v_c - self.v_m * self.v_q);
    if claim_outer_final != outer_final_expected {
      return Err(SpartanError::InvalidSumcheckProof);
    }

    transcript.absorb(
      b"outer_claims",
      &[self.v_a, self.v_b, self.v_c, self.v_m, self.v_q].as_slice(),
    );

    // Inner SC verification.
    let r = transcript.squeeze(b"r")?;
    let claim_inner = self.v_a + r * self.v_b + r * r * self.v_c;

    let (claim_inner_final, r_y) =
      self
        .sc_inner
        .verify(claim_inner, num_rounds_y, 2, &params, &mut transcript)?;

    // Reconstruct eval_Z from eval_W and public IO.
    let eval_x = eval_public_at::<M>(num_rounds_y - 1, &U.x, &r_y[1..], &params);
    let eval_z = (one - r_y[0]) * self.eval_w + r_y[0] * eval_x;

    // Evaluate A, B, C MLEs at (r_x, r_y) via full eq tables.
    let t_x = EqPolynomial::<MScalar<M>>::evals_from_points(&r_x, &params);
    let t_y = EqPolynomial::<MScalar<M>>::evals_from_points(&r_y, &params);
    let (eval_a, eval_b, eval_c) = evaluate_matrices::<M>(shape, &t_x, &t_y, &params);

    let inner_final_expected = (eval_a + r * eval_b + r * r * eval_c) * eval_z;
    if claim_inner_final != inner_final_expected {
      return Err(SpartanError::InvalidSumcheckProof);
    }

    // Mod-PCS verification for W at r_y[1..].
    let comm_eval_w = <ModPCS<M> as ModPCSEngineTrait<M>>::commit(
      &vk.ck_s,
      &[self.eval_w],
      &self.blind_eval_w,
      false,
    )?;
    <ModPCS<M> as ModPCSEngineTrait<M>>::verify(
      &vk.vk_ee,
      &vk.ck_s,
      &mut transcript,
      &U.comm_w,
      &r_y[1..],
      &comm_eval_w,
      &self.eval_arg_w,
    )?;

    // Mod-PCS verification for Q at r_x.
    let comm_eval_q = <ModPCS<M> as ModPCSEngineTrait<M>>::commit(
      &vk.ck_s,
      &[self.v_q],
      &self.blind_eval_q,
      false,
    )?;
    <ModPCS<M> as ModPCSEngineTrait<M>>::verify(
      &vk.vk_ee,
      &vk.ck_s,
      &mut transcript,
      &U.comm_q,
      &r_x,
      &comm_eval_q,
      &self.eval_arg_q,
    )?;

    info!(elapsed_ms = %verify_t.elapsed().as_millis(), "imod_spartan_modp_verify");
    Ok(())
  }
}

// ---------------------------------------------------------------------------
// helpers

/// ABC[y] = sum_i eq_rx[i] · (A[i,y] + r·B[i,y] + r²·C[i,y]),
/// right-padded to length 2·num_vars to match the inner-SC layout.
fn bind_abc<M: ModEngine>(
  shape: &IntModR1CSShapeModp<M>,
  eq_rx: &[MScalar<M>],
  r: &MScalar<M>,
) -> Vec<MScalar<M>> {
  let zero = <MScalar<M> as SumcheckField>::zero(&shape.params);
  let num_cols = shape.num_vars + 1 + shape.num_io;
  let mut abc = vec![zero; num_cols];
  let r_sq = *r * *r;

  for (i, j, val) in &shape.A {
    abc[*j] += eq_rx[*i] * *val;
  }
  for (i, j, val) in &shape.B {
    abc[*j] += *r * eq_rx[*i] * *val;
  }
  for (i, j, val) in &shape.C {
    abc[*j] += r_sq * eq_rx[*i] * *val;
  }
  abc.resize(2 * shape.num_vars, zero);
  abc
}

/// Multilinear extension of the public side `(1, x, 0, …, 0)` evaluated at `r`.
fn eval_public_at<M: ModEngine>(
  num_vars_pub: usize,
  x: &[MScalar<M>],
  r: &[MScalar<M>],
  params: &MParams<M>,
) -> MScalar<M> {
  debug_assert_eq!(r.len(), num_vars_pub);
  let zero = <MScalar<M> as SumcheckField>::zero(params);
  let one = <MScalar<M> as SumcheckField>::one(params);
  let mut pub_vec = Vec::with_capacity(1 << num_vars_pub);
  pub_vec.push(one);
  pub_vec.extend_from_slice(x);
  pub_vec.resize(1 << num_vars_pub, zero);
  dense_evaluate::<M>(&pub_vec, r, params)
}

/// Dense MLE evaluation: sum_k chi_r[k] · z[k].
fn dense_evaluate<M: ModEngine>(
  z: &[MScalar<M>],
  r: &[MScalar<M>],
  params: &MParams<M>,
) -> MScalar<M> {
  let zero = <MScalar<M> as SumcheckField>::zero(params);
  let chis = EqPolynomial::<MScalar<M>>::evals_from_points(r, params);
  debug_assert_eq!(chis.len(), z.len());
  chis
    .par_iter()
    .zip(z.par_iter())
    .map(|(c, v)| *c * *v)
    .reduce(|| zero, |a, b| a + b)
}

/// Evaluate A, B, C MLEs at (r_x, r_y) via precomputed eq-tables.
fn evaluate_matrices<M: ModEngine>(
  shape: &IntModR1CSShapeModp<M>,
  t_x: &[MScalar<M>],
  t_y: &[MScalar<M>],
  params: &MParams<M>,
) -> (MScalar<M>, MScalar<M>, MScalar<M>) {
  let zero = <MScalar<M> as SumcheckField>::zero(params);
  let eval_one = |entries: &Vec<(usize, usize, MScalar<M>)>| -> MScalar<M> {
    entries
      .iter()
      .map(|(i, j, v)| t_x[*i] * t_y[*j] * *v)
      .fold(zero, |a, b| a + b)
  };
  let (eval_a, (eval_b, eval_c)) = rayon::join(
    || eval_one(&shape.A),
    || rayon::join(|| eval_one(&shape.B), || eval_one(&shape.C)),
  );
  (eval_a, eval_b, eval_c)
}

#[cfg(test)]
mod tests {
  use super::*;
  use crate::{
    dyn_prime::DynPrime,
    provider::{T256DynPrimeEngine, pcs::bridge_modpcs::t256_scalar_params},
  };

  type ME = T256DynPrimeEngine;
  type DP = DynPrime<4>;

  /// Toy: prove `a · b ≡ c (mod N)` over the dynamic prime (p = q = T256
  /// scalar prime). One real row + one padding row to make num_cons a power
  /// of two. Witness layout w = [a, b, c, 0].
  fn build_toy(
    a: u64,
    b: u64,
    c: u64,
    n: u64,
    q: u64,
  ) -> (
    IntModR1CSShapeModp<ME>,
    Vec<DP>,
    Vec<DP>,
    crypto_bigint::modular::FixedMontyParams<4>,
  ) {
    let params = t256_scalar_params();
    let num_cons = 2usize;
    let num_vars = 4usize;
    let num_io = 0usize;
    let one = DP::one(&params);

    let mat_a = vec![(0, 0, one)];
    let mat_b = vec![(0, 1, one)];
    let mat_c = vec![(0, 2, one)];
    let mods = vec![DP::from_u64(&params, n), DP::zero(&params)];

    let shape = IntModR1CSShapeModp::<ME>::new(
      num_cons, num_vars, num_io, mat_a, mat_b, mat_c, mods, params,
    )
    .unwrap();

    let w = vec![
      DP::from_u64(&params, a),
      DP::from_u64(&params, b),
      DP::from_u64(&params, c),
      DP::zero(&params),
    ];
    let q_vec = vec![DP::from_u64(&params, q), DP::zero(&params)];
    (shape, w, q_vec, params)
  }

  #[test]
  fn imod_modp_toy_roundtrip() {
    // 3 · 5 = 15 = 1 + 14 · 1
    let (shape, w, q, _params) = build_toy(3, 5, 1, 14, 1);

    let (pk, vk) = IntModSpartanModpSNARK::<ME>::setup(shape.clone()).unwrap();

    let (W, U) = IntModR1CSWitnessModp::<ME>::new(&shape, &pk.ck, w, q, vec![]).unwrap();
    shape.is_sat(&pk.ck, &U, &W).unwrap();

    let proof = IntModSpartanModpSNARK::<ME>::prove(&pk, &U, &W).unwrap();
    proof.verify(&vk, &U).unwrap();
  }

  /// Wrong quotient must be rejected by `is_sat`.
  #[test]
  fn imod_modp_bad_witness_rejected() {
    let (shape, w, _, params) = build_toy(3, 5, 1, 14, 1);
    let bad_q = vec![DP::zero(&params), DP::zero(&params)];
    let (pk, _vk) = IntModSpartanModpSNARK::<ME>::setup(shape.clone()).unwrap();
    let (W, U) = IntModR1CSWitnessModp::<ME>::new(&shape, &pk.ck, w, bad_q, vec![]).unwrap();
    assert!(shape.is_sat(&pk.ck, &U, &W).is_err());
  }

  /// Tampering with v_q must cause verify to reject.
  #[test]
  fn imod_modp_verify_rejects_tampering() {
    let (shape, w, q, params) = build_toy(3, 5, 1, 14, 1);
    let (pk, vk) = IntModSpartanModpSNARK::<ME>::setup(shape.clone()).unwrap();
    let (W, U) = IntModR1CSWitnessModp::<ME>::new(&shape, &pk.ck, w, q, vec![]).unwrap();
    let mut proof = IntModSpartanModpSNARK::<ME>::prove(&pk, &U, &W).unwrap();
    proof.v_q += DP::one(&params);
    assert!(proof.verify(&vk, &U).is_err());
  }
}
