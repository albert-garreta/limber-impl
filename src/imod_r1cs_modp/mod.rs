//! Phase-2 Integer Mod-R1CS relation, parameterized over a `ModEngine`.
//!
//! Mirrors `crate::imod_r1cs` but the scalar field is `M::Scalar`
//! (a `SumcheckField` — e.g. `DynPrime<4>`), not a static-modulus
//! curve scalar. Two structural consequences:
//!
//!   - `M::Scalar` does not implement `ff::PrimeField`, so we cannot
//!     reuse `crate::r1cs::SparseMatrix<F: PrimeField>`. The matrices
//!     are stored as raw `(row, col, val)` COO entries. SpMV is a
//!     straightforward linear walk; tight for the prototype's small
//!     matrices.
//!
//!   - All field constants come from `SumcheckField::{zero,one,from_u64}`
//!     and need the runtime `Params` carried in the shape.
//!
//! Phase 2 invariants follow Phase 1: `num_vars`, `num_cons` are powers
//! of two; `num_vars >= 1 + num_io`; `mods.len() == num_cons`.

use crate::{
  errors::SpartanError,
  traits::mod_engine::{ModEngine, ModPCSEngineTrait, SumcheckEngine, SumcheckField},
};
use rayon::prelude::*;

type MScalar<M> = <M as SumcheckEngine>::Scalar;
type MParams<M> = <MScalar<M> as SumcheckField>::Params;
type ModPCS<M> = <M as ModEngine>::ModPCS;
type ModCK<M> = <ModPCS<M> as ModPCSEngineTrait<M>>::CommitmentKey;
type ModVK<M> = <ModPCS<M> as ModPCSEngineTrait<M>>::VerifierKey;
type ModComm<M> = <ModPCS<M> as ModPCSEngineTrait<M>>::Commitment;
type ModBlind<M> = <ModPCS<M> as ModPCSEngineTrait<M>>::Blind;

/// Phase-2 IntMod-R1CS shape over `M: ModEngine`. Matrices are dense COO
/// triples; `mods` is the per-row modulus vector (paper's `m`); `params`
/// is the runtime modulus context for `M::Scalar`.
#[derive(Clone, Debug)]
pub struct IntModR1CSShapeModp<M: ModEngine> {
  pub(crate) num_cons: usize,
  pub(crate) num_vars: usize,
  pub(crate) num_io: usize,
  pub(crate) A: Vec<(usize, usize, MScalar<M>)>,
  pub(crate) B: Vec<(usize, usize, MScalar<M>)>,
  pub(crate) C: Vec<(usize, usize, MScalar<M>)>,
  pub(crate) mods: Vec<MScalar<M>>,
  pub(crate) params: MParams<M>,
}

/// Witness: assignment `w`, per-row quotients `q`, and the Mod-PCS blinds.
#[derive(Clone, Debug)]
pub struct IntModR1CSWitnessModp<M: ModEngine> {
  pub(crate) w: Vec<MScalar<M>>,
  pub(crate) q: Vec<MScalar<M>>,
  pub(crate) r_w: ModBlind<M>,
  pub(crate) r_q: ModBlind<M>,
}

/// Public instance: witness commitments + public input `x`.
#[derive(Clone, Debug)]
pub struct IntModR1CSInstanceModp<M: ModEngine> {
  pub(crate) comm_w: ModComm<M>,
  pub(crate) comm_q: ModComm<M>,
  pub(crate) x: Vec<MScalar<M>>,
}

impl<M: ModEngine> IntModR1CSShapeModp<M> {
  /// Build a new shape from raw COO entries. Phase-2 invariants:
  /// `num_vars` and `num_cons` are powers of two, `num_vars >= 1 + num_io`,
  /// and `mods.len() == num_cons`.
  pub fn new(
    num_cons: usize,
    num_vars: usize,
    num_io: usize,
    A: Vec<(usize, usize, MScalar<M>)>,
    B: Vec<(usize, usize, MScalar<M>)>,
    C: Vec<(usize, usize, MScalar<M>)>,
    mods: Vec<MScalar<M>>,
    params: MParams<M>,
  ) -> Result<Self, SpartanError> {
    if !num_vars.is_power_of_two() || !num_cons.is_power_of_two() {
      return Err(SpartanError::InvalidInputLength {
        reason: format!(
          "IntModR1CSShapeModp requires power-of-two sizes (got num_vars={num_vars}, num_cons={num_cons})"
        ),
      });
    }
    if num_vars < 1 + num_io {
      return Err(SpartanError::InvalidInputLength {
        reason: format!(
          "num_vars ({num_vars}) must be at least 1 + num_io ({})",
          1 + num_io
        ),
      });
    }
    if mods.len() != num_cons {
      return Err(SpartanError::InvalidInputLength {
        reason: format!(
          "mods length ({}) must equal num_cons ({num_cons})",
          mods.len()
        ),
      });
    }
    let num_cols = num_vars + 1 + num_io;
    for entries in [&A, &B, &C] {
      for (row, col, _) in entries {
        if *row >= num_cons || *col >= num_cols {
          return Err(SpartanError::InvalidIndex);
        }
      }
    }
    Ok(Self {
      num_cons,
      num_vars,
      num_io,
      A,
      B,
      C,
      mods,
      params,
    })
  }

  /// Mod-PCS commitment-key setup. Sized to the larger of `num_vars` /
  /// `num_cons` so a single key can commit either `w` or `q`.
  pub fn commitment_key(&self) -> (ModCK<M>, ModVK<M>) {
    let n = self.num_vars.max(self.num_cons);
    <ModPCS<M> as ModPCSEngineTrait<M>>::setup(b"ck_imod_modp", n, crate::DEFAULT_COMMITMENT_WIDTH)
  }

  /// SpMV: returns `(Az, Bz, Cz)` for `z` of length `num_vars + 1 + num_io`.
  pub fn multiply_vec(
    &self,
    z: &[MScalar<M>],
  ) -> Result<(Vec<MScalar<M>>, Vec<MScalar<M>>, Vec<MScalar<M>>), SpartanError> {
    if z.len() != self.num_vars + 1 + self.num_io {
      return Err(SpartanError::InvalidWitnessLength);
    }
    let zero = <MScalar<M> as SumcheckField>::zero(&self.params);
    let multiply = |entries: &Vec<(usize, usize, MScalar<M>)>| -> Vec<MScalar<M>> {
      let mut out = vec![zero; self.num_cons];
      for (i, j, v) in entries {
        out[*i] += *v * z[*j];
      }
      out
    };
    let (az, (bz, cz)) = rayon::join(
      || multiply(&self.A),
      || rayon::join(|| multiply(&self.B), || multiply(&self.C)),
    );
    Ok((az, bz, cz))
  }

  /// Check satisfaction of `Az ∘ Bz = Cz + m ∘ q` over `M::Scalar`,
  /// and that the commitments open to the claimed `w` and `q`.
  pub fn is_sat(
    &self,
    ck: &ModCK<M>,
    U: &IntModR1CSInstanceModp<M>,
    W: &IntModR1CSWitnessModp<M>,
  ) -> Result<(), SpartanError> {
    if W.w.len() != self.num_vars || W.q.len() != self.num_cons || U.x.len() != self.num_io {
      return Err(SpartanError::InvalidWitnessLength);
    }
    let one = <MScalar<M> as SumcheckField>::one(&self.params);
    let z = [W.w.as_slice(), &[one], U.x.as_slice()].concat();
    let (az, bz, cz) = self.multiply_vec(&z)?;

    let ok_eq = (0..self.num_cons)
      .into_par_iter()
      .all(|i| az[i] * bz[i] == cz[i] + self.mods[i] * W.q[i]);

    let (comm_w_ok, comm_q_ok) = rayon::join(
      || -> Result<bool, SpartanError> {
        let cw = <ModPCS<M> as ModPCSEngineTrait<M>>::commit(ck, &W.w, &W.r_w, false)?;
        Ok(cw == U.comm_w)
      },
      || -> Result<bool, SpartanError> {
        let cq = <ModPCS<M> as ModPCSEngineTrait<M>>::commit(ck, &W.q, &W.r_q, false)?;
        Ok(cq == U.comm_q)
      },
    );
    let comm_w_ok = comm_w_ok?;
    let comm_q_ok = comm_q_ok?;

    if !ok_eq {
      return Err(SpartanError::UnSat {
        reason: "IntMod-R1CS equation does not hold".to_string(),
      });
    }
    if !(comm_w_ok && comm_q_ok) {
      return Err(SpartanError::UnSat {
        reason: "IntMod-R1CS commitment mismatch".to_string(),
      });
    }
    Ok(())
  }

  /// Hash the shape's public data to a 32-byte digest for transcript
  /// binding. Phase-1 reuses `bincode`-based `Digestible`; here we hash
  /// raw bytes because `M::Scalar` (e.g. `DynPrime`) isn't `Serialize`.
  pub fn digest(&self) -> [u8; 32] {
    use sha3::{Digest, Keccak256};
    let mut h = Keccak256::new();
    h.update(b"IntModR1CSShapeModp");
    h.update((self.num_cons as u64).to_le_bytes());
    h.update((self.num_vars as u64).to_le_bytes());
    h.update((self.num_io as u64).to_le_bytes());
    for entries in [&self.A, &self.B, &self.C] {
      h.update((entries.len() as u64).to_le_bytes());
      for (i, j, v) in entries {
        h.update((*i as u64).to_le_bytes());
        h.update((*j as u64).to_le_bytes());
        h.update(v.to_le_bytes());
      }
    }
    h.update((self.mods.len() as u64).to_le_bytes());
    for m in &self.mods {
      h.update(m.to_le_bytes());
    }
    h.finalize().into()
  }
}

impl<M: ModEngine> IntModR1CSWitnessModp<M> {
  /// Commit to `(w, q)` and return the witness/instance pair.
  pub fn new(
    shape: &IntModR1CSShapeModp<M>,
    ck: &ModCK<M>,
    w: Vec<MScalar<M>>,
    q: Vec<MScalar<M>>,
    x: Vec<MScalar<M>>,
  ) -> Result<(Self, IntModR1CSInstanceModp<M>), SpartanError> {
    if w.len() != shape.num_vars || q.len() != shape.num_cons || x.len() != shape.num_io {
      return Err(SpartanError::InvalidWitnessLength);
    }
    let r_w = <ModPCS<M> as ModPCSEngineTrait<M>>::blind(ck, shape.num_vars);
    let r_q = <ModPCS<M> as ModPCSEngineTrait<M>>::blind(ck, shape.num_cons);
    let (comm_w, comm_q) = rayon::join(
      || <ModPCS<M> as ModPCSEngineTrait<M>>::commit(ck, &w, &r_w, false),
      || <ModPCS<M> as ModPCSEngineTrait<M>>::commit(ck, &q, &r_q, false),
    );
    let comm_w = comm_w?;
    let comm_q = comm_q?;
    Ok((
      Self { w, q, r_w, r_q },
      IntModR1CSInstanceModp { comm_w, comm_q, x },
    ))
  }
}
