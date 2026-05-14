//! This module implements the KZH-2 polynomial commitment scheme
use crate::{
  errors::SpartanError,
  math::Math,
  polys::eq::EqPolynomial,
  provider::{
    pcs::hyrax_pc::bind_with_delayed,
    traits::{DlogGroup, DlogGroupExt, PairingGroup},
  },
  traits::{
    Engine, PrimeFieldExt,
    pcs::{CommitmentTrait, PCSEngineTrait},
    transcript::{TranscriptEngineTrait, TranscriptReprTrait},
  },
};
use core::marker::PhantomData;
use digest::{ExtendableOutput, Update};
use ff::{Field, PrimeField};
use rayon::prelude::*;
use serde::{Deserialize, Serialize};
use sha3::Shake256;
use std::io::Read;

/// Provides a commitment engine
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct KZHPCS<E: Engine> {
  _p: PhantomData<E>,
}

/// A type that holds the ck for KZH commitments
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(bound = "")]
pub struct KZHCommitmentKey<E: Engine>
where
  E::GE: DlogGroupExt + PairingGroup,
{
  // in KZH-2, num_rows = num_cols = sqrt(n)
  log_num_rows: usize,                                     // log2(num_rows), ν
  log_num_cols: usize,                                     // log2(num_cols), μ
  H_matrix: Vec<<E::GE as DlogGroup>::AffineGroupElement>, // H^(i,j) = τ_i · G^(j), flat row-major, length 2^(ν+μ)
  a_vec: Vec<<E::GE as DlogGroup>::AffineGroupElement>,    // H^(j)   = α   · G^(j), length 2^μ
}

/// A type that holds the vk for KZH commitments
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(bound = "")]
pub struct KZHVerifierKey<E: Engine>
where
  E::GE: DlogGroupExt + PairingGroup,
{
  log_num_rows: usize,
  log_num_cols: usize,
  a_vec: Vec<<E::GE as DlogGroup>::AffineGroupElement>, // duplicated from ck — verifier needs this for check 2
  v_tau_vec: Vec<<E::GE as PairingGroup>::G2Affine>,    // V^(i) = τ_i · V, length 2^ν
  v_alpha: <E::GE as PairingGroup>::G2Affine,           // V'    = α   · V
}

/// Structure that holds commitments
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(bound = "")]
pub struct KZHCommitment<E: Engine> {
  comm: E::GE,     // the actual commitment C, single G1
  aux: Vec<E::GE>, // row commitments, {D^(x)} cache, length 2^ν
}

type KZHBlind = (); // non-hiding for this slice

/// Provides an implementation of a polynomial evaluation argument
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(bound = "")]
pub struct KZHEvaluationArgument<E: Engine>
where
  E::GE: DlogGroupExt + PairingGroup,
{
  /// f_xL(xR) = f(xL, xR), the row polynomial at fixed xL. Length 2^μ.
  f_xL: Vec<E::Scalar>,
  /// y = f(point) = f_xL(xR), the claimed evaluation value
  y: E::Scalar,
  _p: PhantomData<E>, //placeholder to tie this struct to the Engine type parameter E, needed for the trait impl but not used in the struct fields
}

// helper method for setup to sample the toxic wastes: tau vector and alpha
fn sample_scalars<E: Engine>(
  label: &'static [u8],
  domain: &'static [u8],
  n: usize,
) -> Vec<E::Scalar> {
  let mut shake = Shake256::default();
  shake.update(label);
  shake.update(domain);
  let mut reader = shake.finalize_xof();
  (0..n)
    .map(|_| {
      let mut bytes = [0u8; 64];
      reader.read_exact(&mut bytes).unwrap();
      E::Scalar::from_uniform(&bytes) // PrimeFieldExt method
    })
    .collect()
}

impl<E: Engine> PCSEngineTrait<E> for KZHPCS<E>
where
  E::GE: DlogGroupExt + PairingGroup,
{
  type CommitmentKey = KZHCommitmentKey<E>;
  type VerifierKey = KZHVerifierKey<E>;
  type Commitment = KZHCommitment<E>;
  type Blind = KZHBlind;
  type EvaluationArgument = KZHEvaluationArgument<E>;

  /// Derives ck,vk for KZH PC, where n is the number of evaluations (ie 2 to number of variables) of the polynomial to be committed to, and width is the number of columns of the evaluation point matrix
  fn setup(
    label: &'static [u8],
    _n: usize,
    width: usize,
  ) -> (Self::CommitmentKey, Self::VerifierKey) {
    //validate: n, width are powers of 2; 1 <= width <= n
    assert!(_n.is_power_of_two(), "n must be a power of 2, got {_n}");
    assert!(
      width.is_power_of_two(),
      "width must be a power of 2, got {width}"
    );
    assert!(
      1 <= width && width <= _n,
      "width must be in [1, {_n}], got {width}"
    );

    let num_cols = width;
    let num_rows = _n / num_cols;
    let log_num_rows = num_rows.log_2();
    let log_num_cols = num_cols.log_2();

    // Base generators (transparent, hash-to-curve)
    let gens = E::GE::from_label(label, num_cols); // {G^(j)}_{j∈B_m}, length m
    let v = <E::GE as PairingGroup>::g2_generator();

    // Toxic-waste trapdoors, derived from label via Shake256 → uniform_bytes → Scalar::from_uniform
    let tau = sample_scalars::<E>(label, b"kzh-tau", num_rows); // {τ_i}_i∈B_n
    let alpha = sample_scalars::<E>(label, b"kzh-alpha", 1)[0];

    // H_matrix[i* num_cols + j] = tau_i * g^(j)    for (i, j) in [num_rows] × [num_cols]
    let H_matrix_proj: Vec<E::GE> = (0..num_rows)
      .into_par_iter()
      .flat_map(|i| {
        let tau_i = tau[i];
        gens
          .par_iter()
          .map(move |g_j| E::GE::group(g_j) * tau_i)
          .collect::<Vec<_>>()
      })
      .collect();

    // a_vec[j] = alpha * g^(j)    for j in [num_cols]
    let a_vec_proj: Vec<E::GE> = gens
      .par_iter()
      .map(|g_j| E::GE::group(g_j) * alpha)
      .collect();

    // v_tau_vec[i] = tau_i * v        for i in [num_rows]
    let v_tau_vec_proj: Vec<<E::GE as PairingGroup>::G2> =
      (0..num_rows).into_par_iter().map(|i| v * tau[i]).collect();

    // v_alpha = alpha * v
    let v_alpha_proj = v * alpha;

    //batch-normalize all to affine
    let H_matrix = E::GE::batch_affine(&H_matrix_proj);
    let a_vec = E::GE::batch_affine(&a_vec_proj);
    let v_tau_vec = <E::GE as PairingGroup>::batch_g2_to_affine(&v_tau_vec_proj);
    let v_alpha = <E::GE as PairingGroup>::g2_to_affine(&v_alpha_proj);

    let ck = KZHCommitmentKey {
      log_num_rows,
      log_num_cols,
      H_matrix,
      a_vec,
    };
    let vk = KZHVerifierKey {
      log_num_rows,
      log_num_cols,
      a_vec: ck.a_vec.clone(),
      v_tau_vec,
      v_alpha,
    };
    (ck, vk)
  }

  fn blind(_ck: &Self::CommitmentKey, _n: usize) -> Self::Blind {}

  fn commit(
    ck: &Self::CommitmentKey,
    evals_vec: &[E::Scalar],
    _r: &Self::Blind,
    is_small: bool, //the api trusts the caller to pass honest is_small
  ) -> Result<Self::Commitment, SpartanError> {
    let num_cols = 1usize << ck.log_num_cols;
    let num_rows = 1usize << ck.log_num_rows;

    let expected_len = num_rows * num_cols;
    if evals_vec.len() != expected_len {
      return Err(SpartanError::InvalidInputLength {
        reason: format!(
          "expected evals_vec.len() = {expected_len} (2^{} rows * 2^{} cols), got {}",
          ck.log_num_rows,
          ck.log_num_cols,
          evals_vec.len()
        ),
      });
    }

    // For each row i in B_n in parallel, compute:
    //   C^(i) = MSM(evals_vec[row_i], H_matrix[row_i])  — will be summed together to gfet the overall commitment
    //   D^(i) = MSM(evals_vec[row_i], a_vec)            — cache entry needed by prove/verify
    let row_results: Result<Vec<(E::GE, E::GE)>, SpartanError> = (0..num_rows)
      .into_par_iter()
      .map(|i| {
        let lower = i * num_cols;
        let upper = lower + num_cols;
        let row_scalars = &evals_vec[lower..upper];
        let H_matrix_row = &ck.H_matrix[lower..upper];

        let (C_i, D_i) = if is_small {
          // Caller hints all row scalars fit in u64; take the fast MSM path.
          // Convert once and reuse across both MSMs for this row.
          let small: Vec<u64> = row_scalars
            .iter()
            .map(|s| {
              let bytes = s.to_repr();
              //low 8 bytes -> [u8; 8] -> u64
              u64::from_le_bytes(bytes.as_ref()[..8].try_into().unwrap())
            })
            .collect();
          let C_i = E::GE::vartime_multiscalar_mul_small(&small, H_matrix_row, false)?;
          let D_i = E::GE::vartime_multiscalar_mul_small(&small, &ck.a_vec, false)?;
          (C_i, D_i)
        } else {
          let C_i = E::GE::vartime_multiscalar_mul(row_scalars, H_matrix_row, false)?;
          let D_i = E::GE::vartime_multiscalar_mul(row_scalars, &ck.a_vec, false)?;
          (C_i, D_i)
        };
        Ok((C_i, D_i))
      })
      .collect();
    let row_results = row_results?;

    // C = Σ_i C^(i);  aux = [D^(0), …, D^(num_rows - 1)]
    let mut comm = E::GE::zero();
    let mut aux = Vec::with_capacity(num_rows);
    for (C_i, D_i) in row_results {
      comm += C_i;
      aux.push(D_i);
    }

    Ok(KZHCommitment { comm, aux })
  }

  fn commit_zeros(
    _ck: &Self::CommitmentKey,
    _n: usize,
    _r: &Self::Blind,
  ) -> Result<Self::Commitment, SpartanError> {
    Err(SpartanError::InternalError {
      reason: "not yet implemented".to_string(),
    })
  }

  fn check_commitment(comm: &Self::Commitment, n: usize, width: usize) -> Result<(), SpartanError> {
    if width == 0 || !n.is_multiple_of(width) {
      return Err(SpartanError::InvalidCommitmentLength {
        reason: format!("KZH commitment shape: width {width} must divide n {n}"),
      });
    }
    let expected_num_rows = n / width;
    if comm.aux.len() != expected_num_rows {
      return Err(SpartanError::InvalidCommitmentLength {
        reason: format!(
          "KZH commitment aux length: actual {}, expected {} (n = {n}, width = {width})",
          comm.aux.len(),
          expected_num_rows,
        ),
      });
    }
    Ok(())
  }

  fn rerandomize_commitment(
    _ck: &Self::CommitmentKey,
    _comm: &Self::Commitment,
    _r_old: &Self::Blind,
    _r_new: &Self::Blind,
  ) -> Result<Self::Commitment, SpartanError> {
    Err(SpartanError::InternalError {
      reason: "not yet implemented".to_string(),
    })
  }

  fn combine_commitments(comms: &[Self::Commitment]) -> Result<Self::Commitment, SpartanError> {
    if comms.is_empty() {
      return Err(SpartanError::InvalidInputLength {
        reason: "KZH combine_commitments: no commitments provided".to_string(),
      });
    }
    // Concatenate the per-row aux caches and sum the comm fields. For length-1
    // input this is the identity. For multi-partial inputs this is correct when
    // the partials are slices of a single witness committed under a compatible
    // shared SRS (zero-padded outside each partial's row range); independently
    // sampled SRSes would not produce a meaningful sum on the comm field.

    let combined_comm: E::GE = comms
      .iter()
      .map(|c| c.comm)
      .fold(E::GE::zero(), |acc, c| acc + c);
    let combined_aux: Vec<E::GE> = comms.iter().flat_map(|c| c.aux.iter().cloned()).collect();
    Ok(KZHCommitment {
      comm: combined_comm,
      aux: combined_aux,
    })
  }

  fn combine_blinds(_blinds: &[Self::Blind]) -> Result<Self::Blind, SpartanError> {
    // Non-hiding: Blind is ().
    Ok(())
  }

  fn prove(
    ck: &Self::CommitmentKey,
    _ck_eval: &Self::CommitmentKey,
    transcript: &mut <E as Engine>::TE,
    comm: &Self::Commitment,
    evals_vec: &[<E as Engine>::Scalar],
    _blind: &Self::Blind,
    point: &[<E as Engine>::Scalar],
    _comm_eval: &Self::Commitment,
    _blind_eval: &Self::Blind,
  ) -> Result<Self::EvaluationArgument, SpartanError> {
    let log_num_rows = ck.log_num_rows;
    let log_num_cols = ck.log_num_cols;
    let num_rows = 1usize << log_num_rows;
    let num_cols = 1usize << log_num_cols;

    // number of variables does not match
    if point.len() != log_num_rows + log_num_cols {
      return Err(SpartanError::InvalidInputLength {
        reason: format!(
          "KZH prove: point has length {}, expected {}",
          point.len(),
          log_num_rows + log_num_cols
        ),
      });
    }
    if evals_vec.len() != num_rows * num_cols {
      return Err(SpartanError::InvalidInputLength {
        reason: format!(
          "KZH prove: evals_vec has length {}, expected {}",
          evals_vec.len(),
          num_rows * num_cols
        ),
      });
    }

    // Bind FS transcript to (commitment, point) before producing the proof.
    transcript.absorb(b"poly_com", comm);
    transcript.absorb(b"point", &point);

    // points = (xL,xR) split to left and right halves
    // f_xL(xR) = f(xL, xR) = Σ_i eq(i, xL) · f(xL, xR).
    // Layout: evals_vec is row-major, evals_vec[xL · num_cols + xR] = f(xL, xR).
    // bind_with_delayed computes the vector-matrix product with delayed Montgomery
    // reduction (~50% faster than a naive double loop).
    let eq_l = EqPolynomial::new(point[..log_num_rows].to_vec()).evals();
    let f_xL = bind_with_delayed(evals_vec, &eq_l, num_cols);

    // y = f(point) = f_xL(xR) = <f_xL, eq(xR)>.
    let eq_r = EqPolynomial::new(point[log_num_rows..].to_vec()).evals();
    let y: E::Scalar = f_xL
      .iter()
      .zip(eq_r.iter())
      .map(|(a, b)| *a * *b)
      .fold(E::Scalar::ZERO, |acc, x| acc + x);

    transcript.absorb(b"f_xL", &f_xL.as_slice());
    transcript.absorb(b"y", &y);

    Ok(KZHEvaluationArgument {
      f_xL,
      y,
      _p: PhantomData,
    })
  }

  fn verify(
    vk: &Self::VerifierKey,
    ck_eval: &Self::CommitmentKey,
    transcript: &mut <E as Engine>::TE,
    comm: &Self::Commitment,
    point: &[<E as Engine>::Scalar],
    comm_eval: &Self::Commitment,
    arg: &Self::EvaluationArgument,
  ) -> Result<(), SpartanError> {
    let log_num_rows = vk.log_num_rows;
    let log_num_cols = vk.log_num_cols;
    let num_rows = 1usize << log_num_rows;
    let num_cols = 1usize << log_num_cols;

    // number of variables does not match
    if point.len() != log_num_rows + log_num_cols {
      return Err(SpartanError::InvalidInputLength {
        reason: format!(
          "KZH verify: point has length {}, expected {}",
          point.len(),
          log_num_rows + log_num_cols
        ),
      });
    }
    if arg.f_xL.len() != num_cols {
      return Err(SpartanError::InvalidInputLength {
        reason: format!(
          "KZH verify: f_xL has length {}, expected {}",
          arg.f_xL.len(),
          num_cols
        ),
      });
    }
    if comm.aux.len() != num_rows {
      return Err(SpartanError::InvalidCommitmentLength {
        reason: format!(
          "KZH verify: comm.aux has length {}, expected {}",
          comm.aux.len(),
          num_rows
        ),
      });
    }

    // Bind FS transcript identically to the prover.
    transcript.absorb(b"poly_com", comm);
    transcript.absorb(b"point", &point);
    transcript.absorb(b"f_xL", &arg.f_xL.as_slice());
    transcript.absorb(b"y", &arg.y);

    let (xL, xR) = point.split_at(log_num_rows);

    // Check 1 (pairing): e(C, v alpha) == Σ_i e(D^(i), v tau^(i))
    let C_aff = E::GE::affine(&comm.comm);
    let D_affs: Vec<_> = comm.aux.iter().map(E::GE::affine).collect();
    let lhs1 = <E::GE as PairingGroup>::multi_pairing(&[(&C_aff, &vk.v_alpha)]);
    let rhs1_pairs: Vec<_> = D_affs.iter().zip(vk.v_tau_vec.iter()).collect();
    let rhs1 = <E::GE as PairingGroup>::multi_pairing(&rhs1_pairs);
    if lhs1 != rhs1 {
      return Err(SpartanError::ProofVerifyError {
        reason: "KZH verify: pairing check failed (e(C, V') != Π e(D^(x), V^(x)))".to_string(),
      });
    }

    // Check 2 (vector identity): Σ_i f_xL[i] · H^(i) == Σ_i eq(i, xL) · D^(i)
    let eq_l = EqPolynomial::new(xL.to_vec()).evals();
    let lhs2 = E::GE::vartime_multiscalar_mul(&arg.f_xL, &vk.a_vec, false)?;
    let rhs2 = E::GE::vartime_multiscalar_mul(&eq_l, &D_affs, false)?;
    if lhs2 != rhs2 {
      return Err(SpartanError::ProofVerifyError {
        reason: "KZH verify: vector identity check failed".to_string(),
      });
    }

    // Check 3a (paper's check (iii)): arg.y = f_xL(xR).
    let eq_r = EqPolynomial::new(xR.to_vec()).evals();
    let f_xL_at_xR: E::Scalar = arg
      .f_xL
      .iter()
      .zip(eq_r.iter())
      .map(|(a, b)| *a * *b)
      .fold(E::Scalar::ZERO, |acc, x| acc + x);
    if f_xL_at_xR != arg.y {
      return Err(SpartanError::ProofVerifyError {
        reason: "KZH verify: claimed evaluation y inconsistent with f_xL(xR)".to_string(),
      });
    }

    // Check 3b (API compliance): comm_eval must commit to arg.y.
    // this is not needed for KZH but needed for Spartan soundness....
    let expected_comm_eval = Self::commit(ck_eval, &[arg.y], &(), false)?;
    if expected_comm_eval != *comm_eval {
      return Err(SpartanError::ProofVerifyError {
        reason: "KZH verify: comm_eval inconsistent with claimed evaluation y".to_string(),
      });
    }

    Ok(())
  }
}

impl<E: Engine> TranscriptReprTrait<E::GE> for KZHCommitment<E>
where
  E::GE: DlogGroupExt,
{
  fn to_transcript_bytes(&self) -> Vec<u8> {
    self.comm.to_transcript_bytes()
  }
}

impl<E: Engine> CommitmentTrait<E> for KZHCommitment<E> where E::GE: DlogGroupExt {}

#[cfg(test)]
mod tests {
  use super::*;
  use crate::provider::Bn254Engine;
  use ff::Field;
  use rand_core::OsRng;

  type E = Bn254Engine;
  type GE = <E as Engine>::GE;
  type Scalar = <E as Engine>::Scalar;

  #[test]
  fn test_setup_shapes() {
    // Balanced split: 64 = 8 × 8  (ν = μ = 3)
    let (ck, vk) = KZHPCS::<E>::setup(b"kzh_setup_balanced", 64, 8);
    assert_eq!(ck.log_num_rows, 3);
    assert_eq!(ck.log_num_cols, 3);
    assert_eq!(ck.H_matrix.len(), 64);
    assert_eq!(ck.a_vec.len(), 8);
    assert_eq!(vk.a_vec.len(), 8);
    assert_eq!(vk.v_tau_vec.len(), 8);

    // Unbalanced split: 64 = 4 × 16  (ν = 2, μ = 4)
    let (ck, vk) = KZHPCS::<E>::setup(b"kzh_setup_unbalanced", 64, 16);
    assert_eq!(ck.log_num_rows, 2);
    assert_eq!(ck.log_num_cols, 4);
    assert_eq!(ck.H_matrix.len(), 64);
    assert_eq!(ck.a_vec.len(), 16);
    assert_eq!(vk.a_vec.len(), 16);
    assert_eq!(vk.v_tau_vec.len(), 4);
  }

  #[test]
  fn test_commit_shapes() {
    let (ck, _vk) = KZHPCS::<E>::setup(b"kzh_commit_shapes", 64, 8);
    let mut rng = OsRng;

    // Random polynomial of size 64
    let v: Vec<Scalar> = (0..64).map(|_| Scalar::random(&mut rng)).collect();
    let comm = KZHPCS::<E>::commit(&ck, &v, &(), false).expect("commit");

    // Aux must have one entry per row (num_rows = 64 / 8 = 8).
    assert_eq!(comm.aux.len(), 8);
    // C should not be the identity for a random polynomial.
    assert_ne!(
      comm.comm,
      GE::zero(),
      "C is identity for a random polynomial"
    );

    // is_small path agreement: build a polynomial whose scalars are genuinely
    // small (fit in u64), commit both ways, results must be identical.
    let small_v: Vec<Scalar> = (0..64u64).map(|i| Scalar::from(i % 7)).collect();
    let c_small = KZHPCS::<E>::commit(&ck, &small_v, &(), true).expect("commit small");
    let c_reg = KZHPCS::<E>::commit(&ck, &small_v, &(), false).expect("commit reg");
    assert_eq!(c_small, c_reg, "is_small path diverges from regular path");

    // Wrong-length input must error, not panic.
    let bad = vec![Scalar::ZERO; 63];
    assert!(
      KZHPCS::<E>::commit(&ck, &bad, &(), false).is_err(),
      "commit accepted a vector of wrong length"
    );

    // check_commitment shape validation: correct shape Ok, mismatched shape Err.
    assert!(KZHPCS::<E>::check_commitment(&comm, 64, 8).is_ok());
    assert!(KZHPCS::<E>::check_commitment(&comm, 64, 16).is_err()); // wrong width
    assert!(KZHPCS::<E>::check_commitment(&comm, 128, 8).is_err()); // wrong n
  }

  #[test]
  fn test_commit_pairing_consistency() {
    // The load-bearing structural check:
    //     e(C, V') == ∏_i e(D^(i), V^(i))
    // If setup and commit are internally consistent, this holds for any polynomial.
    let (ck, vk) = KZHPCS::<E>::setup(b"kzh_pairing_consistency", 64, 8);

    let mut rng = OsRng;
    let v: Vec<Scalar> = (0..64).map(|_| Scalar::random(&mut rng)).collect();
    let comm = KZHPCS::<E>::commit(&ck, &v, &(), false).expect("commit");

    let C_aff = GE::affine(&comm.comm);
    let D_affs: Vec<_> = comm.aux.iter().map(GE::affine).collect();

    // LHS: e(C, V')
    let lhs = <GE as PairingGroup>::multi_pairing(&[(&C_aff, &vk.v_alpha)]);

    // RHS: ∏_i e(D^(i), V^(i))
    let rhs_pairs: Vec<_> = D_affs.iter().zip(vk.v_tau_vec.iter()).collect();
    let rhs = <GE as PairingGroup>::multi_pairing(&rhs_pairs);

    assert_eq!(
      lhs, rhs,
      "KZH pairing consistency failed: e(C, V') != ∏ e(D^(i), V^(i))"
    );
  }

  use crate::provider::keccak::Keccak256Transcript;
  use crate::traits::transcript::TranscriptEngineTrait;

  /// Ground-truth multilinear evaluation: ⟨poly, eq(point)⟩.
  fn poly_eval(poly: &[Scalar], point: &[Scalar]) -> Scalar {
    let eq_pt = EqPolynomial::new(point.to_vec()).evals();
    poly
      .iter()
      .zip(eq_pt.iter())
      .map(|(a, b)| *a * *b)
      .fold(Scalar::ZERO, |acc, x| acc + x)
  }

  /// Helper: full prove/verify round-trip with given (n, width).
  fn run_round_trip(n: usize, width: usize) {
    let (ck, vk) = KZHPCS::<E>::setup(b"kzh_pv_main", n, width);
    let (ck_eval, _vk_eval) = KZHPCS::<E>::setup(b"kzh_pv_eval", 1, 1);

    let log_n = (n as f64).log2() as usize;
    let mut rng = OsRng;
    let poly: Vec<Scalar> = (0..n).map(|_| Scalar::random(&mut rng)).collect();
    let point: Vec<Scalar> = (0..log_n).map(|_| Scalar::random(&mut rng)).collect();

    // Independent evaluation of poly at point — this is what comm_eval commits to.
    let y = poly_eval(&poly, &point);

    let comm = KZHPCS::<E>::commit(&ck, &poly, &(), false).expect("commit");
    let comm_eval = KZHPCS::<E>::commit(&ck_eval, &[y], &(), false).expect("commit_eval");

    let mut prover_ts: Keccak256Transcript<E> = Keccak256Transcript::new(b"pv_test");
    let arg = KZHPCS::<E>::prove(
      &ck,
      &ck_eval,
      &mut prover_ts,
      &comm,
      &poly,
      &(),
      &point,
      &comm_eval,
      &(),
    )
    .expect("prove");

    let mut verifier_ts: Keccak256Transcript<E> = Keccak256Transcript::new(b"pv_test");
    KZHPCS::<E>::verify(
      &vk,
      &ck_eval,
      &mut verifier_ts,
      &comm,
      &point,
      &comm_eval,
      &arg,
    )
    .expect("verify");
  }

  #[test]
  fn test_prove_verify_round_trip_balanced() {
    run_round_trip(64, 8); // ν = μ = 3
  }

  #[test]
  fn test_prove_verify_round_trip_unbalanced() {
    run_round_trip(64, 16); // ν = 2, μ = 4
  }

  #[test]
  fn test_verify_rejects_tampered_f_xL() {
    let (ck, vk) = KZHPCS::<E>::setup(b"kzh_tamper_main", 64, 8);
    let (ck_eval, _) = KZHPCS::<E>::setup(b"kzh_tamper_eval", 1, 1);

    let mut rng = OsRng;
    let poly: Vec<Scalar> = (0..64).map(|_| Scalar::random(&mut rng)).collect();
    let point: Vec<Scalar> = (0..6).map(|_| Scalar::random(&mut rng)).collect();
    let y = poly_eval(&poly, &point);

    let comm = KZHPCS::<E>::commit(&ck, &poly, &(), false).unwrap();
    let comm_eval = KZHPCS::<E>::commit(&ck_eval, &[y], &(), false).unwrap();

    let mut p_ts: Keccak256Transcript<E> = Keccak256Transcript::new(b"tamper");
    let mut arg = KZHPCS::<E>::prove(
      &ck,
      &ck_eval,
      &mut p_ts,
      &comm,
      &poly,
      &(),
      &point,
      &comm_eval,
      &(),
    )
    .unwrap();

    // Flip one entry in f_xL.
    arg.f_xL[0] += Scalar::ONE;

    let mut v_ts: Keccak256Transcript<E> = Keccak256Transcript::new(b"tamper");
    let result = KZHPCS::<E>::verify(&vk, &ck_eval, &mut v_ts, &comm, &point, &comm_eval, &arg);
    assert!(result.is_err(), "tampered f_xL must be rejected, got Ok");
  }

  #[test]
  fn test_verify_rejects_wrong_comm_eval() {
    let (ck, vk) = KZHPCS::<E>::setup(b"kzh_wrong_eval_main", 64, 8);
    let (ck_eval, _) = KZHPCS::<E>::setup(b"kzh_wrong_eval_eval", 1, 1);

    let mut rng = OsRng;
    let poly: Vec<Scalar> = (0..64).map(|_| Scalar::random(&mut rng)).collect();
    let point: Vec<Scalar> = (0..6).map(|_| Scalar::random(&mut rng)).collect();

    // Commit_eval commits to a DIFFERENT value, not the true eval.
    let wrong_z = Scalar::random(&mut rng);
    let comm = KZHPCS::<E>::commit(&ck, &poly, &(), false).unwrap();
    let comm_eval = KZHPCS::<E>::commit(&ck_eval, &[wrong_z], &(), false).unwrap();

    let mut p_ts: Keccak256Transcript<E> = Keccak256Transcript::new(b"wrong_eval");
    let arg = KZHPCS::<E>::prove(
      &ck,
      &ck_eval,
      &mut p_ts,
      &comm,
      &poly,
      &(),
      &point,
      &comm_eval,
      &(),
    )
    .unwrap();

    let mut v_ts: Keccak256Transcript<E> = Keccak256Transcript::new(b"wrong_eval");
    let result = KZHPCS::<E>::verify(&vk, &ck_eval, &mut v_ts, &comm, &point, &comm_eval, &arg);
    assert!(
      result.is_err(),
      "mismatched comm_eval must be rejected, got Ok"
    );
  }
}
