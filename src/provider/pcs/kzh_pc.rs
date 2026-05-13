//! This module implements the KZH-2 polynomial commitment scheme
use crate::{
  errors::SpartanError, math::Math, provider::traits::{DlogGroup, DlogGroupExt, PairingGroup}, traits::{
    Engine, PrimeFieldExt,
    pcs::{CommitmentTrait, PCSEngineTrait},
    transcript::TranscriptReprTrait,
  }
};
use core::marker::PhantomData;
use digest::{ExtendableOutput, Update};
use ff::PrimeField;
use rayon::prelude::*;
use serde::{Deserialize, Serialize};
use sha3::Shake256;
use std::io::Read;


/// Provides a commitment engine
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct KZHPCS<E:Engine> {
    _p: PhantomData<E>
}

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
      E::Scalar::from_uniform(&bytes)  // PrimeFieldExt method
    })
    .collect()
}

/// A type that holds commitment generators for KZH commitments
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(bound = "")]
pub struct KZHCommitmentKey<E:Engine>
where 
  E::GE: DlogGroupExt + PairingGroup,
{
  log_num_rows: usize,                                // log2(num_rows)
  log_num_cols: usize,                                // log2(num_cols)
  h_matrix: Vec<<E::GE as DlogGroup>::AffineGroupElement>, // H^(i,j) = τ_i · G^(j), flat row-major, length 2^(ν+μ)
  h_col:    Vec<<E::GE as DlogGroup>::AffineGroupElement>, // H^(j)   = α   · G^(j), length 2^μ
}

/// A type that holds the verifier key for KZH commitments
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(bound = "")]
pub struct KZHVerifierKey<E:Engine>
where
  E::GE: DlogGroupExt + PairingGroup,
{
  log_num_rows: usize,
  log_num_cols: usize,
  h_col:   Vec<<E::GE as DlogGroup>::AffineGroupElement>,  // duplicated from CK — verifier needs this for check 2
  v_row:   Vec<<E::GE as PairingGroup>::G2Affine>, // V^(i) = τ_i · V, length 2^ν
  v_prime: <E::GE as PairingGroup>::G2Affine,      // V'    = α   · V
}

/// Structure that holds commitments
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(bound = "")]
pub struct KZHCommitment<E:Engine> {
  comm:   E::GE,        // the actual commitment, single G1
  aux: Vec<E::GE>,   // {D^(x)} cache, length 2^ν
}

type KZHBlind = ();          // non-hiding for this slice

/// Provides an implementation of a polynomial evaluation argument
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(bound = "")]
pub struct KZHEvaluationArgument<E:Engine>
where
  E::GE: DlogGroupExt + PairingGroup,
{  
  _p:PhantomData<E> // placeholder; populated in next slice
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

  /// Derives generators for KZH PC, where n is the size of the vector to be committed to and width is the number of columns.
  fn setup(
    label: &'static [u8],
    _n: usize,
    width: usize,
  ) -> (Self::CommitmentKey, Self::VerifierKey) {
    //validate: n, width are powers of 2; 1 <= width <= n
    assert!(_n.is_power_of_two(), "n must be a power of 2, got {_n}");
    assert!(width.is_power_of_two(), "width must be a power of 2, got {width}");
    assert!(1 <= width && width <= _n, "width must be in [1, {_n}], got {width}");

    let num_cols = width;
    let num_rows = _n / num_cols;
    let log_num_rows = num_rows.log_2();
    let log_num_cols = num_cols.log_2();

    // Base generators (transparent, hash-to-curve)
    let gens  = E::GE::from_label(label, num_cols);                 // {G^(j)}_j∈B_m, length m
    let V    = <E::GE as PairingGroup>::g2_generator();


    // Toxic-waste trapdoors, derived from label via Shake256 → uniform_bytes → Scalar::from_uniform
    let tau   = sample_scalars::<E>(label, b"kzh-tau",   num_rows);  // {τ_i}_i∈B_n
    let alpha    = sample_scalars::<E>(label, b"kzh-alpha", 1)[0];


    // Derive structured SRS

    // H_matrix[i* num_cols + j] = tau_i * G^(j)    for (i, j) in [num_rows] × [num_cols]
    let h_matrix_proj: Vec<E::GE> = (0..num_rows)
      .into_par_iter()
      .flat_map(|i| {
      let tau_i = tau[i];
      gens.par_iter().map(move |g_j| E::GE::group(g_j) * tau_i).collect::<Vec<_>>()
    }).collect();

    //H_col[j] = alpha * G^(j)    for j in [num_cols]   
    let h_col_proj: Vec<E::GE> = gens
      .par_iter()
      .map(|g_j| E::GE::group(g_j) * alpha)
      .collect();

    //V_row[i]  = tau_i * V        for i in [num_rows]             
    let v_row_proj: Vec<<E::GE as PairingGroup>::G2> = (0..num_rows)
      .into_par_iter()
      .map(|i| V * tau[i])
      .collect();

    // V_prime = alpha * V
    let v_prime_proj = V*alpha;

    //batch-normalize all to affine
    let h_matrix = E::GE::batch_affine(&h_matrix_proj);
    let h_col = E::GE::batch_affine(&h_col_proj);
    let v_row = <E::GE as PairingGroup>::batch_g2_to_affine(&v_row_proj);
    let v_prime = <E::GE as PairingGroup>::g2_to_affine(&v_prime_proj);

    let ck = KZHCommitmentKey { log_num_rows, log_num_cols, h_matrix, h_col };
    let vk = KZHVerifierKey { log_num_rows, log_num_cols, h_col: ck.h_col.clone(), v_row, v_prime };
    return (ck, vk)
  }

  fn blind(_ck: &Self::CommitmentKey, _n: usize) -> Self::Blind { () }


  fn commit(
    ck: &Self::CommitmentKey,
    v: &[E::Scalar],
    _r: &Self::Blind,
    is_small: bool, //the api trusts the caller to pass honest is_small
  ) -> Result<Self::Commitment, SpartanError> {
    let num_cols = 1usize << ck.log_num_cols;
    let num_rows = 1usize << ck.log_num_rows;
    let expected_len = num_rows * num_cols;

    if v.len() != expected_len {
        return Err(SpartanError::InvalidInputLength {
        reason: format!(
            "expected v.len() = {expected_len} (2^{} rows * 2^{} cols), got {}",
            ck.log_num_rows,
            ck.log_num_cols,
            v.len()
        ),
        });
    }

    // For each row i in B_n in parallel, compute:
    //   C^(i) = MSM(v[row_i], h_matrix[row_i])  — contribution to the overall commitment
    //   D^(i) = MSM(v[row_i], h_col)            — cache entry needed by prove/verify
    let row_results: Result<Vec<(E::GE, E::GE)>, SpartanError> = (0..num_rows)
        .into_par_iter()
        .map(|i| {
        let lower = i * num_cols;
        let upper = lower + num_cols;
        let row_scalars = &v[lower..upper];
        let h_matrix_row = &ck.h_matrix[lower..upper];

        let (c_i, d_i) = if is_small {
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
            let c_i = E::GE::vartime_multiscalar_mul_small(&small, h_matrix_row, false)?;
            let d_i = E::GE::vartime_multiscalar_mul_small(&small, &ck.h_col, false)?;
            (c_i, d_i)
        } else {
            let c_i = E::GE::vartime_multiscalar_mul(row_scalars, h_matrix_row, false)?;
            let d_i = E::GE::vartime_multiscalar_mul(row_scalars, &ck.h_col, false)?;
            (c_i, d_i)
        };
        Ok((c_i, d_i))
        })
        .collect();
    let row_results = row_results?;

    // C = Σ_i C^(i);  aux = [D^(0), …, D^(num_rows - 1)]
    let mut comm = E::GE::zero();
    let mut aux = Vec::with_capacity(num_rows);
    for (c_i, d_i) in row_results {
        comm = comm + c_i;
        aux.push(d_i);
    }

    Ok(KZHCommitment { comm, aux })
}


  fn commit_zeros(
      _ck: &Self::CommitmentKey,
      _n: usize,
      _r: &Self::Blind,
    ) -> Result<Self::Commitment, SpartanError>
  {
    Err(SpartanError::InternalError { reason: "not yet implemented".to_string() })
  }

  fn check_commitment(comm: &Self::Commitment, n: usize, width: usize) -> Result<(), SpartanError> {
    if width == 0 || n % width != 0 {
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
    ) -> Result<Self::Commitment, SpartanError>
  {
    Err(SpartanError::InternalError { reason: "not yet implemented".to_string() })
  }

  fn combine_commitments(_comms: &[Self::Commitment]) -> Result<Self::Commitment, SpartanError> {
    Err(SpartanError::InternalError { reason: "not yet implemented".to_string() })
  }

  fn combine_blinds(_blinds: &[Self::Blind]) -> Result<Self::Blind, SpartanError> {
    Err(SpartanError::InternalError { reason: "not yet implemented".to_string() })
  }

  fn prove(
      _ck: &Self::CommitmentKey,
      _ck_eval: &Self::CommitmentKey,
      _transcript: &mut <E as Engine>::TE,
      _comm: &Self::Commitment,
      _poly: &[<E as Engine>::Scalar],
      _blind: &Self::Blind,
      _point: &[<E as Engine>::Scalar],
      _comm_eval: &Self::Commitment,
      _blind_eval: &Self::Blind,
    ) -> Result<Self::EvaluationArgument, SpartanError>
  {
    Err(SpartanError::InternalError { reason: "not yet implemented".to_string() })
  }

  fn verify(
      _vk: &Self::VerifierKey,
      _ck_eval: &Self::CommitmentKey,
      _transcript: &mut <E as Engine>::TE,
      _comm: &Self::Commitment,
      _point: &[<E as Engine>::Scalar],
      _comm_eval: &Self::Commitment,
      _arg: &Self::EvaluationArgument,
    ) -> Result<(), SpartanError>
  {
    Err(SpartanError::InternalError { reason: "not yet implemented".to_string() })
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
    assert_eq!(ck.h_matrix.len(), 64);
    assert_eq!(ck.h_col.len(), 8);
    assert_eq!(vk.h_col.len(), 8);
    assert_eq!(vk.v_row.len(), 8);

    // Unbalanced split: 64 = 4 × 16  (ν = 2, μ = 4)
    let (ck, vk) = KZHPCS::<E>::setup(b"kzh_setup_unbalanced", 64, 16);
    assert_eq!(ck.log_num_rows, 2);
    assert_eq!(ck.log_num_cols, 4);
    assert_eq!(ck.h_matrix.len(), 64);
    assert_eq!(ck.h_col.len(), 16);
    assert_eq!(vk.h_col.len(), 16);
    assert_eq!(vk.v_row.len(), 4);
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
    assert_ne!(comm.comm, GE::zero(), "C is identity for a random polynomial");

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

    let c_aff = GE::affine(&comm.comm);
    let d_affs: Vec<_> = comm.aux.iter().map(|d| GE::affine(d)).collect();

    // LHS: e(C, V')
    let lhs = <GE as PairingGroup>::multi_pairing(&[(&c_aff, &vk.v_prime)]);

    // RHS: ∏_i e(D^(i), V^(i))
    let rhs_pairs: Vec<_> = d_affs.iter().zip(vk.v_row.iter()).collect();
    let rhs = <GE as PairingGroup>::multi_pairing(&rhs_pairs);

    assert_eq!(
      lhs, rhs,
      "KZH pairing consistency failed: e(C, V') != ∏ e(D^(i), V^(i))"
    );
  }
}
