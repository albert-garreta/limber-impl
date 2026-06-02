//! Phase-2 Integer Mod-R1CS SNARK driver, generic over `M: ModEngine`.
//!
//! Mirrors `crate::imod_spartan` but the shape, witness, and matrix
//! entries are integer-valued (`BigUint`), the prime `p` over which the
//! sumcheck runs is sampled from the transcript via Fiat-Shamir (Miller-
//! Rabin rejection sampling, see `M::sample_params`), and the SNARK
//! verifies the IntMod-R1CS relation `Az ∘ Bz = Cz + m ∘ q` mod that
//! sampled `p`. The Mod-PCS commits integer polynomials and opens at
//! `Z_p` points returning `Z_p` evals — the `p ≠ q` reconciliation is
//! the Mod-PCS's responsibility (Phase-3 IntEval); the Phase-2 driver
//! treats it as a black-box contract.
//!
//! Flow:
//!   1. Bootstrap transcript with `M::bootstrap_params()` (placeholder).
//!   2. Byte-absorb the vk digest, two integer-poly commitments, and the
//!      public IO `x` (`BigUint` LE bytes).
//!   3. `params = M::sample_params(transcript)` derives `p` from squeeze
//!      bytes; `transcript.set_params(params)` switches typed-squeeze
//!      reductions into `Z_p`.
//!   4. Reduce shape/witness/IO `BigUint`s to `M::Scalar` mod `p`.
//!   5. Run outer + inner sumchecks in `Z_p`.
//!   6. Open `w` and `q` at `Z_p` points via `M::ModPCS::prove`, passing
//!      the original `BigUint` polynomials (integer view).
//!
//! Single witness segment; no shared/precommitted/rest split; no limb
//! decomposition; no range checks; no BDDT first-round optimization.

use crate::{
  errors::SpartanError,
  imod_r1cs_modp::{IntModR1CSInstanceModp, IntModR1CSShapeModp, IntModR1CSWitnessModp},
  math::Math,
  polys_modp::{eq::EqPolynomial, multilinear::MultilinearPolynomial},
  provider::keccak::Keccak256Transcript,
  start_span,
  sumcheck_modp::SumcheckProof,
  traits::{
    mod_engine::{ModEngine, ModPCSEngineTrait, SumcheckEngine, SumcheckField},
    transcript::{ByteTranscript, TranscriptEngineTrait},
  },
};
use num_bigint::BigUint;
use rayon::prelude::*;
use tracing::info;

type MScalar<M> = <M as SumcheckEngine>::Scalar;
type MParams<M> = <MScalar<M> as SumcheckField>::Params;
type ModPCS<M> = <M as ModEngine>::ModPCS;
type ModCK<M> = <ModPCS<M> as ModPCSEngineTrait<M>>::CommitmentKey;
type ModVK<M> = <ModPCS<M> as ModPCSEngineTrait<M>>::VerifierKey;
type ModBlind<M> = <ModPCS<M> as ModPCSEngineTrait<M>>::Blind;
type ModEvalArg<M> = <ModPCS<M> as ModPCSEngineTrait<M>>::EvaluationArgument;

/// Convert a `BigUint` integer into an `M::Scalar` value by reducing
/// modulo the runtime modulus carried in `params`.
fn biguint_to_scalar<M: ModEngine>(v: &BigUint, params: &MParams<M>) -> MScalar<M> {
  MScalar::<M>::from_bytes_reduce(params, &v.to_bytes_le())
}

fn biguint_vec_to_scalars<M: ModEngine>(v: &[BigUint], params: &MParams<M>) -> Vec<MScalar<M>> {
  v.par_iter()
    .map(|b| biguint_to_scalar::<M>(b, params))
    .collect()
}

fn biguint_matrix_to_scalars<M: ModEngine>(
  entries: &[(usize, usize, BigUint)],
  params: &MParams<M>,
) -> Vec<(usize, usize, MScalar<M>)> {
  entries
    .par_iter()
    .map(|(i, j, v)| (*i, *j, biguint_to_scalar::<M>(v, params)))
    .collect()
}

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
/// dynamic-prime types (`M::Scalar`, `Params`) aren't `Serialize` yet.
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

impl<M> IntModSpartanModpSNARK<M>
where
  M: ModEngine<TE = Keccak256Transcript<M>>,
{
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

    // 1. Bootstrap transcript with placeholder params.
    let mut transcript =
      Keccak256Transcript::<M>::new_with_params(b"IntModSpartanModpSNARK", M::bootstrap_params());

    // 2. Byte-absorb pre-`p` data. Public IO is `BigUint`, not a
    //    `TranscriptReprTrait` type, so absorb its LE bytes directly.
    transcript.absorb_bytes(b"vk", &pk.vk_digest);
    transcript.absorb(b"comm_w", &U.comm_w);
    transcript.absorb(b"comm_q", &U.comm_q);
    for xi in &U.x {
      transcript.absorb_bytes(b"x", &xi.to_bytes_le());
    }

    // 3. Sample `p` from the transcript and switch typed-squeeze context.
    let params = M::sample_params(&mut transcript);
    transcript.set_params(params.clone());

    let shape = &pk.shape;
    let num_vars = shape.num_vars;
    let num_cons = shape.num_cons;
    let num_rounds_x = num_cons.log_2();
    let num_rounds_y = num_vars.log_2() + 1;

    let zero = MScalar::<M>::zero(&params);
    let one = MScalar::<M>::one(&params);

    // 4. Reduce shape/witness/IO from BigUint to M::Scalar mod p.
    let (_red_span, red_t) = start_span!("imod_modp_reduce");
    let mods_p = biguint_vec_to_scalars::<M>(&shape.mods, &params);
    let w_p = biguint_vec_to_scalars::<M>(&W.w, &params);
    let q_p = biguint_vec_to_scalars::<M>(&W.q, &params);
    let x_p = biguint_vec_to_scalars::<M>(&U.x, &params);
    let a_p = biguint_matrix_to_scalars::<M>(&shape.A, &params);
    let b_p = biguint_matrix_to_scalars::<M>(&shape.B, &params);
    let c_p = biguint_matrix_to_scalars::<M>(&shape.C, &params);
    info!(elapsed_ms = %red_t.elapsed().as_millis(), "imod_modp_reduce");

    // z = (W, 1, X), padded to 2*num_vars for the MLE.
    let mut z = Vec::with_capacity(2 * num_vars);
    z.extend_from_slice(&w_p);
    z.push(one);
    z.extend_from_slice(&x_p);
    z.resize(2 * num_vars, zero);

    let z_for_spmv = &z[..num_vars + 1 + shape.num_io];
    let (az, bz, cz) = spmv::<M>(&a_p, &b_p, &c_p, z_for_spmv, num_cons, &params);

    // Outer sumcheck: sum_i eq(i, tau) · (Az·Bz − Cz − M·Q) = 0.
    let tau: Vec<MScalar<M>> = (0..num_rounds_x)
      .map(|_| transcript.squeeze(b"tau"))
      .collect::<Result<Vec<_>, SpartanError>>()?;

    let mut poly_az = MultilinearPolynomial::new(az, params.clone());
    let mut poly_bz = MultilinearPolynomial::new(bz, params.clone());
    let mut poly_cz = MultilinearPolynomial::new(cz, params.clone());
    let mut poly_m = MultilinearPolynomial::new(mods_p, params.clone());
    let mut poly_q = MultilinearPolynomial::new(q_p.clone(), params.clone());

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
    let abc = bind_abc::<M>(
      &a_p,
      &b_p,
      &c_p,
      num_vars,
      shape.num_io,
      &eq_rx,
      &r,
      &params,
    );

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
    let eval_x = eval_public_at::<M>(num_rounds_y - 1, &x_p, &r_y[1..], &params);
    let one_minus_r0 = one - r_y[0];
    let inv = one_minus_r0.invert().ok_or(SpartanError::DivisionByZero)?;
    let eval_w = (eval_z - r_y[0] * eval_x) * inv;

    // Mod-PCS open W at r_y[1..]. Mod-PCS commits/opens integers — pass
    // the original BigUint witness and the Z_p eval reduced into a
    // BigUint in [0, p).
    let (_wopen_span, wopen_t) = start_span!("imod_modp_w_open");
    let eval_w_bu = BigUint::from_bytes_le(&eval_w.to_le_bytes());
    let blind_eval_w = <ModPCS<M> as ModPCSEngineTrait<M>>::blind(&pk.ck_s, 1);
    let comm_eval_w = <ModPCS<M> as ModPCSEngineTrait<M>>::commit(
      &pk.ck_s,
      std::slice::from_ref(&eval_w_bu),
      &blind_eval_w,
      false,
    )?;
    let eval_arg_w = <ModPCS<M> as ModPCSEngineTrait<M>>::prove(
      &pk.ck,
      &pk.ck_s,
      &mut transcript,
      &U.comm_w,
      &W.w,
      &W.r_w,
      &r_y[1..],
      &eval_w_bu,
      &comm_eval_w,
      &blind_eval_w,
    )?;
    info!(elapsed_ms = %wopen_t.elapsed().as_millis(), "imod_modp_w_open");

    // Mod-PCS open Q at r_x.
    let (_qopen_span, qopen_t) = start_span!("imod_modp_q_open");
    let v_q_bu = BigUint::from_bytes_le(&v_q.to_le_bytes());
    let blind_eval_q = <ModPCS<M> as ModPCSEngineTrait<M>>::blind(&pk.ck_s, 1);
    let comm_eval_q = <ModPCS<M> as ModPCSEngineTrait<M>>::commit(
      &pk.ck_s,
      std::slice::from_ref(&v_q_bu),
      &blind_eval_q,
      false,
    )?;
    let eval_arg_q = <ModPCS<M> as ModPCSEngineTrait<M>>::prove(
      &pk.ck,
      &pk.ck_s,
      &mut transcript,
      &U.comm_q,
      &W.q,
      &W.r_q,
      &r_x,
      &v_q_bu,
      &comm_eval_q,
      &blind_eval_q,
    )?;
    info!(elapsed_ms = %qopen_t.elapsed().as_millis(), "imod_modp_q_open");

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

    // 1. Bootstrap transcript identically to prove().
    let mut transcript =
      Keccak256Transcript::<M>::new_with_params(b"IntModSpartanModpSNARK", M::bootstrap_params());

    // 2. Byte-absorb pre-`p` data identically to prove().
    transcript.absorb_bytes(b"vk", &vk.digest);
    transcript.absorb(b"comm_w", &U.comm_w);
    transcript.absorb(b"comm_q", &U.comm_q);
    for xi in &U.x {
      transcript.absorb_bytes(b"x", &xi.to_bytes_le());
    }

    // 3. Re-sample `p` from the same byte stream → identical params.
    let params = M::sample_params(&mut transcript);
    transcript.set_params(params.clone());

    let shape = &vk.shape;
    let num_vars = shape.num_vars;
    let num_cons = shape.num_cons;
    let num_rounds_x = num_cons.log_2();
    let num_rounds_y = num_vars.log_2() + 1;

    let zero = MScalar::<M>::zero(&params);
    let one = MScalar::<M>::one(&params);

    // 4. Reduce shape/IO from BigUint to M::Scalar mod p.
    let mods_p = biguint_vec_to_scalars::<M>(&shape.mods, &params);
    let x_p = biguint_vec_to_scalars::<M>(&U.x, &params);
    let a_p = biguint_matrix_to_scalars::<M>(&shape.A, &params);
    let b_p = biguint_matrix_to_scalars::<M>(&shape.B, &params);
    let c_p = biguint_matrix_to_scalars::<M>(&shape.C, &params);

    // Outer SC verification.
    let tau: Vec<MScalar<M>> = (0..num_rounds_x)
      .map(|_| transcript.squeeze(b"tau"))
      .collect::<Result<Vec<_>, SpartanError>>()?;

    let (claim_outer_final, r_x) =
      self
        .sc_outer
        .verify(zero, num_rounds_x, 3, &params, &mut transcript)?;

    // v_m matches the public mods MLE at r_x.
    let v_m_expected = dense_evaluate::<M>(&mods_p, &r_x, &params);
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
    let eval_x = eval_public_at::<M>(num_rounds_y - 1, &x_p, &r_y[1..], &params);
    let eval_z = (one - r_y[0]) * self.eval_w + r_y[0] * eval_x;

    // Evaluate A, B, C MLEs at (r_x, r_y) via full eq tables.
    let t_x = EqPolynomial::<MScalar<M>>::evals_from_points(&r_x, &params);
    let t_y = EqPolynomial::<MScalar<M>>::evals_from_points(&r_y, &params);
    let (eval_a, eval_b, eval_c) = evaluate_matrices::<M>(&a_p, &b_p, &c_p, &t_x, &t_y, &params);

    let inner_final_expected = (eval_a + r * eval_b + r * r * eval_c) * eval_z;
    if claim_inner_final != inner_final_expected {
      return Err(SpartanError::InvalidSumcheckProof);
    }

    // Mod-PCS verification for W at r_y[1..].
    let (_wver_span, wver_t) = start_span!("imod_modp_w_verify");
    let eval_w_bu = BigUint::from_bytes_le(&self.eval_w.to_le_bytes());
    let comm_eval_w = <ModPCS<M> as ModPCSEngineTrait<M>>::commit(
      &vk.ck_s,
      std::slice::from_ref(&eval_w_bu),
      &self.blind_eval_w,
      false,
    )?;
    <ModPCS<M> as ModPCSEngineTrait<M>>::verify(
      &vk.vk_ee,
      &vk.ck_s,
      &mut transcript,
      &U.comm_w,
      &r_y[1..],
      &eval_w_bu,
      &comm_eval_w,
      &self.eval_arg_w,
    )?;
    info!(elapsed_ms = %wver_t.elapsed().as_millis(), "imod_modp_w_verify");

    // Mod-PCS verification for Q at r_x.
    let (_qver_span, qver_t) = start_span!("imod_modp_q_verify");
    let v_q_bu = BigUint::from_bytes_le(&self.v_q.to_le_bytes());
    let comm_eval_q = <ModPCS<M> as ModPCSEngineTrait<M>>::commit(
      &vk.ck_s,
      std::slice::from_ref(&v_q_bu),
      &self.blind_eval_q,
      false,
    )?;
    <ModPCS<M> as ModPCSEngineTrait<M>>::verify(
      &vk.vk_ee,
      &vk.ck_s,
      &mut transcript,
      &U.comm_q,
      &r_x,
      &v_q_bu,
      &comm_eval_q,
      &self.eval_arg_q,
    )?;
    info!(elapsed_ms = %qver_t.elapsed().as_millis(), "imod_modp_q_verify");

    info!(elapsed_ms = %verify_t.elapsed().as_millis(), "imod_spartan_modp_verify");
    Ok(())
  }
}

// ---------------------------------------------------------------------------
// helpers (operate on pre-reduced M::Scalar matrices)

fn spmv<M: ModEngine>(
  a: &[(usize, usize, MScalar<M>)],
  b: &[(usize, usize, MScalar<M>)],
  c: &[(usize, usize, MScalar<M>)],
  z: &[MScalar<M>],
  num_cons: usize,
  params: &MParams<M>,
) -> (Vec<MScalar<M>>, Vec<MScalar<M>>, Vec<MScalar<M>>) {
  let zero = MScalar::<M>::zero(params);
  let multiply = |entries: &[(usize, usize, MScalar<M>)]| -> Vec<MScalar<M>> {
    let mut out = vec![zero; num_cons];
    for (i, j, v) in entries {
      out[*i] += *v * z[*j];
    }
    out
  };
  let (az, (bz, cz)) = rayon::join(
    || multiply(a),
    || rayon::join(|| multiply(b), || multiply(c)),
  );
  (az, bz, cz)
}

/// ABC[y] = sum_i eq_rx[i] · (A[i,y] + r·B[i,y] + r²·C[i,y]),
/// right-padded to length 2·num_vars to match the inner-SC layout.
fn bind_abc<M: ModEngine>(
  a: &[(usize, usize, MScalar<M>)],
  b: &[(usize, usize, MScalar<M>)],
  c: &[(usize, usize, MScalar<M>)],
  num_vars: usize,
  num_io: usize,
  eq_rx: &[MScalar<M>],
  r: &MScalar<M>,
  params: &MParams<M>,
) -> Vec<MScalar<M>> {
  let zero = MScalar::<M>::zero(params);
  let num_cols = num_vars + 1 + num_io;
  let mut abc = vec![zero; num_cols];
  let r_sq = *r * *r;

  for (i, j, val) in a {
    abc[*j] += eq_rx[*i] * *val;
  }
  for (i, j, val) in b {
    abc[*j] += *r * eq_rx[*i] * *val;
  }
  for (i, j, val) in c {
    abc[*j] += r_sq * eq_rx[*i] * *val;
  }
  abc.resize(2 * num_vars, zero);
  abc
}

/// Multilinear extension of the public side `(1, x, 0, …, 0)` evaluated at `r`.
fn eval_public_at<M: ModEngine>(
  num_vars_pub: usize,
  x_p: &[MScalar<M>],
  r: &[MScalar<M>],
  params: &MParams<M>,
) -> MScalar<M> {
  debug_assert_eq!(r.len(), num_vars_pub);
  let zero = MScalar::<M>::zero(params);
  let one = MScalar::<M>::one(params);
  let mut pub_vec = Vec::with_capacity(1 << num_vars_pub);
  pub_vec.push(one);
  pub_vec.extend_from_slice(x_p);
  pub_vec.resize(1 << num_vars_pub, zero);
  dense_evaluate::<M>(&pub_vec, r, params)
}

/// Dense MLE evaluation: sum_k chi_r[k] · z[k].
fn dense_evaluate<M: ModEngine>(
  z: &[MScalar<M>],
  r: &[MScalar<M>],
  params: &MParams<M>,
) -> MScalar<M> {
  let zero = MScalar::<M>::zero(params);
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
  a: &[(usize, usize, MScalar<M>)],
  b: &[(usize, usize, MScalar<M>)],
  c: &[(usize, usize, MScalar<M>)],
  t_x: &[MScalar<M>],
  t_y: &[MScalar<M>],
  params: &MParams<M>,
) -> (MScalar<M>, MScalar<M>, MScalar<M>) {
  let zero = MScalar::<M>::zero(params);
  let eval_one = |entries: &[(usize, usize, MScalar<M>)]| -> MScalar<M> {
    entries
      .iter()
      .map(|(i, j, v)| t_x[*i] * t_y[*j] * *v)
      .fold(zero, |a, b| a + b)
  };
  let (eval_a, (eval_b, eval_c)) = rayon::join(
    || eval_one(a),
    || rayon::join(|| eval_one(b), || eval_one(c)),
  );
  (eval_a, eval_b, eval_c)
}

#[cfg(test)]
mod tests {
  use super::*;
  use crate::provider::T256DynPrimeEngine;

  type ME = T256DynPrimeEngine;

  /// Toy: prove `a · b ≡ c (mod N)` over an arbitrary verifier-sampled
  /// prime `p`. Witness layout `w = [a, b, c, 0]`. One real row + one
  /// padding row to make `num_cons` a power of two.
  fn build_toy(
    a: u64,
    b: u64,
    c: u64,
    n: u64,
    q: u64,
  ) -> (IntModR1CSShapeModp<ME>, Vec<BigUint>, Vec<BigUint>) {
    let num_cons = 2usize;
    let num_vars = 4usize;
    let num_io = 0usize;
    let one = BigUint::from(1u32);
    let zero = BigUint::from(0u32);

    let mat_a = vec![(0, 0, one.clone())];
    let mat_b = vec![(0, 1, one.clone())];
    let mat_c = vec![(0, 2, one)];
    let mods = vec![BigUint::from(n), zero.clone()];

    let shape =
      IntModR1CSShapeModp::<ME>::new(num_cons, num_vars, num_io, mat_a, mat_b, mat_c, mods)
        .unwrap();

    let w = vec![
      BigUint::from(a),
      BigUint::from(b),
      BigUint::from(c),
      zero.clone(),
    ];
    let q_vec = vec![BigUint::from(q), zero];
    (shape, w, q_vec)
  }

  /// End-to-end: 3 · 5 ≡ 1 (mod 14) under a transcript-sampled prime `p`
  /// that is *not* the curve scalar prime. Validates the dual-field
  /// driver flow against the trivial Mod-PCS stub.
  #[test]
  fn imod_modp_toy_roundtrip() {
    let (shape, w, q) = build_toy(3, 5, 1, 14, 1);
    let (pk, vk) = IntModSpartanModpSNARK::<ME>::setup(shape.clone()).unwrap();
    let (W, U) = IntModR1CSWitnessModp::<ME>::new(&shape, &pk.ck, w, q, vec![]).unwrap();
    shape.is_sat(&pk.ck, &U, &W).unwrap();
    let proof = IntModSpartanModpSNARK::<ME>::prove(&pk, &U, &W).unwrap();
    proof.verify(&vk, &U).unwrap();
  }

  /// `is_sat` rejects an inconsistent witness (wrong quotient).
  #[test]
  fn imod_modp_bad_witness_rejected() {
    let (shape, w, _) = build_toy(3, 5, 1, 14, 1);
    let bad_q = vec![BigUint::from(0u32), BigUint::from(0u32)];
    let (pk, _vk) = IntModSpartanModpSNARK::<ME>::setup(shape.clone()).unwrap();
    let (W, U) = IntModR1CSWitnessModp::<ME>::new(&shape, &pk.ck, w, bad_q, vec![]).unwrap();
    assert!(shape.is_sat(&pk.ck, &U, &W).is_err());
  }

  /// Tampering with `v_q` breaks transcript binding inside the SNARK
  /// driver (independent of Mod-PCS soundness).
  #[test]
  fn imod_modp_verify_rejects_tampering() {
    let (shape, w, q) = build_toy(3, 5, 1, 14, 1);
    let (pk, vk) = IntModSpartanModpSNARK::<ME>::setup(shape.clone()).unwrap();
    let (W, U) = IntModR1CSWitnessModp::<ME>::new(&shape, &pk.ck, w, q, vec![]).unwrap();
    let mut proof = IntModSpartanModpSNARK::<ME>::prove(&pk, &U, &W).unwrap();
    // The sampled p is engine-dependent; use M::Scalar::one() against the
    // tampered v_q by re-running the same sampling deterministically.
    let mut t = Keccak256Transcript::<ME>::new_with_params(
      b"IntModSpartanModpSNARK",
      <ME as ModEngine>::bootstrap_params(),
    );
    t.absorb_bytes(b"vk", &pk.vk_digest);
    t.absorb(b"comm_w", &U.comm_w);
    t.absorb(b"comm_q", &U.comm_q);
    let params = <ME as ModEngine>::sample_params(&mut t);
    proof.v_q += MScalar::<ME>::one(&params);
    assert!(proof.verify(&vk, &U).is_err());
  }

  /// Two real constraints with different moduli, exercising the outer SC
  /// on more than one active row.
  ///   row 0: 3·5 ≡ 1 (mod 14), q₀ = 1 (since 15 = 1 + 14·1)
  ///   row 1: 7·9 ≡ 3 (mod 20), q₁ = 3 (since 63 = 3 + 20·3)
  #[test]
  fn imod_modp_two_row_roundtrip() {
    let one = BigUint::from(1u32);
    let zero = BigUint::from(0u32);
    let num_cons = 2usize;
    let num_vars = 8usize;
    let num_io = 0usize;

    // Layout w = [a1, b1, c1, a2, b2, c2, 0, 0]
    let mat_a = vec![(0, 0, one.clone()), (1, 3, one.clone())];
    let mat_b = vec![(0, 1, one.clone()), (1, 4, one.clone())];
    let mat_c = vec![(0, 2, one.clone()), (1, 5, one)];
    let mods = vec![BigUint::from(14u32), BigUint::from(20u32)];

    let shape =
      IntModR1CSShapeModp::<ME>::new(num_cons, num_vars, num_io, mat_a, mat_b, mat_c, mods)
        .unwrap();

    let w: Vec<BigUint> = [3u32, 5, 1, 7, 9, 3, 0, 0]
      .iter()
      .map(|x| BigUint::from(*x))
      .collect();
    let q: Vec<BigUint> = [1u32, 3].iter().map(|x| BigUint::from(*x)).collect();

    let (pk, vk) = IntModSpartanModpSNARK::<ME>::setup(shape.clone()).unwrap();
    let (W, U) = IntModR1CSWitnessModp::<ME>::new(&shape, &pk.ck, w, q, vec![]).unwrap();
    shape.is_sat(&pk.ck, &U, &W).unwrap();

    let _ = zero; // silence unused if test layout changes
    let proof = IntModSpartanModpSNARK::<ME>::prove(&pk, &U, &W).unwrap();
    proof.verify(&vk, &U).unwrap();
  }

  /// End-to-end SNARK roundtrip that triggers the IntEval partial-eval
  /// iteration path (step C) on the W open. With `num_vars = 256`, the
  /// Mod-PCS opens W at a point of length `log_2(256) = 8 > k = 7`
  /// (default `IntEvalParams`), so `t = 1` partial-eval iteration runs
  /// per small prime. The Q open at length 1 still uses the step-B
  /// path (no iteration). Both must agree end-to-end through the
  /// SNARK protocol.
  ///
  /// The smallest available SNARK trigger for step C — `num_vars = 128`
  /// gives `point.len = 7 = k`, exactly at the no-iteration boundary,
  /// so we go one power of two above to be sure.
  #[test]
  fn imod_modp_snark_with_inteval_iteration() {
    let one = BigUint::from(1u32);
    let zero = BigUint::from(0u32);
    let num_cons = 2usize;
    let num_vars = 256usize; // log_2(256) = 8 > default k = 7
    let num_io = 0usize;

    // One real row: 3·5 ≡ 1 (mod 14), q₀ = 1. Layout: w[0..3] = [3, 5, 1],
    // rest zero (253 trailing zeros).
    let mat_a = vec![(0, 0, one.clone())];
    let mat_b = vec![(0, 1, one.clone())];
    let mat_c = vec![(0, 2, one)];
    let mods = vec![BigUint::from(14u32), zero.clone()];

    let shape =
      IntModR1CSShapeModp::<ME>::new(num_cons, num_vars, num_io, mat_a, mat_b, mat_c, mods)
        .unwrap();

    let mut w: Vec<BigUint> = vec![zero.clone(); num_vars];
    w[0] = BigUint::from(3u32);
    w[1] = BigUint::from(5u32);
    w[2] = BigUint::from(1u32);
    let q: Vec<BigUint> = vec![BigUint::from(1u32), zero];

    let (pk, vk) = IntModSpartanModpSNARK::<ME>::setup(shape.clone()).unwrap();
    let (W, U) = IntModR1CSWitnessModp::<ME>::new(&shape, &pk.ck, w, q, vec![]).unwrap();
    shape.is_sat(&pk.ck, &U, &W).unwrap();

    let proof = IntModSpartanModpSNARK::<ME>::prove(&pk, &U, &W).unwrap();
    proof.verify(&vk, &U).unwrap();
  }

  /// Public IO: a tiny circuit `w₀ · w₁ ≡ x₀ (mod 14)` with `x₀ = 1` as
  /// public input. Exercises the `eval_public_at` path that the
  /// zero-IO toy doesn't touch.
  #[test]
  fn imod_modp_with_public_io() {
    let one = BigUint::from(1u32);
    let num_cons = 2usize;
    let num_vars = 4usize;
    let num_io = 1usize;
    // num_cols = num_vars + 1 + num_io = 4 + 1 + 1 = 6
    // columns 0..4 = w, column 4 = 1 (constant), column 5 = x[0]
    let mat_a = vec![(0, 0, one.clone())]; // selects w[0] = 3
    let mat_b = vec![(0, 1, one.clone())]; // selects w[1] = 5
    let mat_c = vec![(0, 5, one)]; // selects x[0] = 1
    let mods = vec![BigUint::from(14u32), BigUint::from(0u32)];

    let shape =
      IntModR1CSShapeModp::<ME>::new(num_cons, num_vars, num_io, mat_a, mat_b, mat_c, mods)
        .unwrap();

    let w: Vec<BigUint> = [3u32, 5, 0, 0].iter().map(|x| BigUint::from(*x)).collect();
    let q: Vec<BigUint> = [1u32, 0].iter().map(|x| BigUint::from(*x)).collect();
    let x: Vec<BigUint> = vec![BigUint::from(1u32)];

    let (pk, vk) = IntModSpartanModpSNARK::<ME>::setup(shape.clone()).unwrap();
    let (W, U) = IntModR1CSWitnessModp::<ME>::new(&shape, &pk.ck, w, q, x).unwrap();
    shape.is_sat(&pk.ck, &U, &W).unwrap();

    let proof = IntModSpartanModpSNARK::<ME>::prove(&pk, &U, &W).unwrap();
    proof.verify(&vk, &U).unwrap();
  }

  /// Shapes with the same dimensions/mods but different `A` entries must
  /// produce distinct verifier-key digests. Distinct digests are what
  /// makes vk-cross-binding work: the transcript binds `vk_digest` first,
  /// so swapping vks deterministically derives different `p` and the
  /// rest of the proof becomes incoherent under the wrong vk.
  ///
  /// (Phase-2 gap: today, verifying a proof under the wrong vk
  /// *panics* inside `crypto-bigint`'s `FixedMontyForm` op when the
  /// proof's `DynPrime` values carry params from the original `p` while
  /// the verifier reduces shape2 data with the freshly sampled `p`.
  /// The panic IS a form of rejection, but it's ungraceful — Phase 3
  /// should convert the param mismatch into a clean `SpartanError`.
  /// For now this test asserts only the digest distinction.)
  #[test]
  fn imod_modp_digest_binds_matrices() {
    let one = BigUint::from(1u32);
    let two = BigUint::from(2u32);
    let zero = BigUint::from(0u32);
    let num_cons = 2usize;
    let num_vars = 4usize;
    let num_io = 0usize;

    let mat_a1 = vec![(0, 0, one.clone())];
    let mat_a2 = vec![(0, 0, two)];
    let mat_b = vec![(0, 1, one.clone())];
    let mat_c = vec![(0, 2, one)];
    let mods = vec![BigUint::from(14u32), zero];

    let shape1 = IntModR1CSShapeModp::<ME>::new(
      num_cons,
      num_vars,
      num_io,
      mat_a1,
      mat_b.clone(),
      mat_c.clone(),
      mods.clone(),
    )
    .unwrap();
    let shape2 =
      IntModR1CSShapeModp::<ME>::new(num_cons, num_vars, num_io, mat_a2, mat_b, mat_c, mods)
        .unwrap();

    let (_, vk1) = IntModSpartanModpSNARK::<ME>::setup(shape1).unwrap();
    let (_, vk2) = IntModSpartanModpSNARK::<ME>::setup(shape2).unwrap();
    assert_ne!(vk1.digest(), vk2.digest());
  }

  /// Sanity: the transcript-sampled `p` actually differs from the curve
  /// scalar prime `q`. Asserts the dual-field claim is real on this
  /// engine; the sampling derives `p` from the transcript bytes, so this
  /// also pins the byte-level Fiat-Shamir derivation in place.
  #[test]
  fn imod_modp_sampled_p_is_not_q() {
    use crate::provider::t256_scalar_params;
    let (shape, w, q) = build_toy(3, 5, 1, 14, 1);
    let (pk, _vk) = IntModSpartanModpSNARK::<ME>::setup(shape.clone()).unwrap();
    let (W, U) = IntModR1CSWitnessModp::<ME>::new(&shape, &pk.ck, w, q, vec![]).unwrap();
    let _ = W; // not needed for this check

    let mut t = Keccak256Transcript::<ME>::new_with_params(
      b"IntModSpartanModpSNARK",
      <ME as ModEngine>::bootstrap_params(),
    );
    t.absorb_bytes(b"vk", &pk.vk_digest);
    t.absorb(b"comm_w", &U.comm_w);
    t.absorb(b"comm_q", &U.comm_q);
    let params_p = <ME as ModEngine>::sample_params(&mut t);
    let params_q = t256_scalar_params();
    // Compare the modulus values via the Montgomery context's stable
    // representation: convert 1 to canonical Uint via `retrieve()`, then
    // compare moduli through MontyForm's `params()` accessor.
    // (`FixedMontyParams` doesn't impl PartialEq; route through DynPrime.)
    use crate::dyn_prime::DynPrime;
    let one_p = DynPrime::<4>::one(&params_p);
    let one_q = DynPrime::<4>::one(&params_q);
    assert_ne!(one_p.params(), one_q.params());
  }
}
