// TODO Phase 2 step 6+: remove `allow(dead_code)` once a Phase-2
// IntModSpartan driver starts consuming this type.
#![allow(dead_code)]

//! Dynamic-modulus prime field `DynPrime<LIMBS>`, backed by
//! `crypto_bigint::modular::FixedMontyForm<LIMBS>`. The runtime modulus
//! is sampled by the verifier and carried inside each value via its
//! `FixedMontyParams<LIMBS>` context.
//!
//! Used by Phase-2 IntMod-Spartan when the SNARK arithmetic happens
//! modulo a verifier-sampled ~128-bit prime that isn't known at compile
//! time. See `docs/imod_r1cs_plan.md` for the broader plan.

use crate::traits::{mod_engine::SumcheckField, transcript::TranscriptReprTrait};
use crypto_bigint::{
  Uint,
  modular::{FixedMontyForm, FixedMontyParams},
};
use std::ops::{Add, AddAssign, Mul, MulAssign, Neg, Sub, SubAssign};

/// A field element in a runtime-modulus prime field, stored in Montgomery
/// form. The modulus (and the Montgomery constants derived from it) lives
/// inside the wrapped `FixedMontyForm`'s `FixedMontyParams`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct DynPrime<const LIMBS: usize> {
  inner: FixedMontyForm<LIMBS>,
}

impl<const LIMBS: usize> DynPrime<LIMBS> {
  /// Build from a canonical (non-Montgomery) `Uint` value and the modulus
  /// context. The constructor performs the Montgomery conversion.
  pub fn new(value: Uint<LIMBS>, params: &FixedMontyParams<LIMBS>) -> Self {
    Self {
      inner: FixedMontyForm::new(&value, params),
    }
  }

  /// Wrap an already-Montgomery-form `FixedMontyForm` directly.
  pub(crate) fn from_inner(inner: FixedMontyForm<LIMBS>) -> Self {
    Self { inner }
  }

  /// Extract the canonical (non-Montgomery) representation.
  pub fn retrieve(&self) -> Uint<LIMBS> {
    self.inner.retrieve()
  }

  /// Reference to the modulus context carried by this value.
  pub fn params(&self) -> &FixedMontyParams<LIMBS> {
    self.inner.params()
  }
}

// ---- Arithmetic ops -------------------------------------------------------

impl<const LIMBS: usize> Add for DynPrime<LIMBS> {
  type Output = Self;
  fn add(self, rhs: Self) -> Self {
    Self::from_inner(self.inner + rhs.inner)
  }
}

impl<const LIMBS: usize> Sub for DynPrime<LIMBS> {
  type Output = Self;
  fn sub(self, rhs: Self) -> Self {
    Self::from_inner(self.inner - rhs.inner)
  }
}

impl<const LIMBS: usize> Mul for DynPrime<LIMBS> {
  type Output = Self;
  fn mul(self, rhs: Self) -> Self {
    Self::from_inner(self.inner * rhs.inner)
  }
}

impl<const LIMBS: usize> Neg for DynPrime<LIMBS> {
  type Output = Self;
  fn neg(self) -> Self {
    Self::from_inner(-self.inner)
  }
}

impl<const LIMBS: usize> AddAssign for DynPrime<LIMBS> {
  fn add_assign(&mut self, rhs: Self) {
    self.inner += rhs.inner;
  }
}

impl<const LIMBS: usize> SubAssign for DynPrime<LIMBS> {
  fn sub_assign(&mut self, rhs: Self) {
    self.inner -= rhs.inner;
  }
}

impl<const LIMBS: usize> MulAssign for DynPrime<LIMBS> {
  fn mul_assign(&mut self, rhs: Self) {
    self.inner *= rhs.inner;
  }
}

// ---- SumcheckField --------------------------------------------------------

impl<const LIMBS: usize> SumcheckField for DynPrime<LIMBS> {
  type Params = FixedMontyParams<LIMBS>;

  fn zero(params: &Self::Params) -> Self {
    Self::from_inner(FixedMontyForm::zero(params))
  }

  fn one(params: &Self::Params) -> Self {
    Self::from_inner(FixedMontyForm::one(params))
  }

  fn from_u64(params: &Self::Params, v: u64) -> Self {
    Self::new(Uint::from(v), params)
  }

  fn invert(&self) -> Option<Self> {
    Option::<FixedMontyForm<LIMBS>>::from(self.inner.invert()).map(Self::from_inner)
  }

  fn to_le_bytes(&self) -> Vec<u8> {
    self.retrieve().to_le_bytes().as_slice().to_vec()
  }

  fn from_bytes_reduce(params: &Self::Params, bytes: &[u8]) -> Self {
    let n = Uint::<LIMBS>::BYTES;
    if bytes.len() <= n {
      let mut buf = vec![0u8; n];
      buf[..bytes.len()].copy_from_slice(bytes);
      let value = Uint::<LIMBS>::from_le_slice(&buf);
      return Self::new(value, params);
    }
    // Value exceeds LIMBS limbs — Horner reduction over n-byte chunks.
    // V = c_0 + c_1·B + c_2·B² + …  where B = 2^(n·8).
    let two_32 = Self::from_u64(params, 1u64 << 32);
    let two_64 = two_32 * two_32;
    let mut shift = Self::one(params);
    for _ in 0..LIMBS {
      shift *= two_64;
    }
    let num_chunks = bytes.len().div_ceil(n);
    let mut acc = Self::zero(params);
    for ci in (0..num_chunks).rev() {
      let start = ci * n;
      let end = (start + n).min(bytes.len());
      let mut buf = vec![0u8; n];
      buf[..end - start].copy_from_slice(&bytes[start..end]);
      let chunk_value = Uint::<LIMBS>::from_le_slice(&buf);
      acc = acc * shift + Self::new(chunk_value, params);
    }
    acc
  }
}

// Absorbing a DynPrime into a ByteTranscript: hand over its canonical LE
// bytes. The Phase-2 SNARK driver needs this to absorb mods, IO, and the
// outer-SC claims into the shared Fiat-Shamir chain.
impl<const LIMBS: usize> TranscriptReprTrait for DynPrime<LIMBS> {
  fn to_transcript_bytes(&self) -> Vec<u8> {
    self.to_le_bytes()
  }
}

#[cfg(test)]
mod tests {
  use super::*;
  use crypto_bigint::{Odd, U256};

  #[test]
  #[ignore]
  fn width_microbench() {
    use crypto_bigint::U128;
    use std::time::Instant;
    // Same 128-bit prime (top bit set) in both carriers.
    let m128: U128 = U128::from_be_hex("ffffffffffffffffffffffffffffff61");
    let m256: U256 =
      U256::from_be_hex("00000000000000000000000000000000ffffffffffffffffffffffffffffff61");
    let p2 = FixedMontyParams::new(Odd::new(m128).unwrap());
    let p4 = FixedMontyParams::new(Odd::new(m256).unwrap());
    const ITERS: usize = 10_000_000;
    let mut a2 = DynPrime::<2>::new(U128::from(0x1234_5678_9abc_def0_u64), &p2);
    let b2 = DynPrime::<2>::new(U128::from(0x0fed_cba9_8765_4321_u64), &p2);
    let t = Instant::now();
    for _ in 0..ITERS {
      a2 *= b2;
    }
    let ns2 = t.elapsed().as_nanos() as f64 / ITERS as f64;
    std::hint::black_box(a2);
    let mut a4 = DynPrime::<4>::new(U256::from(0x1234_5678_9abc_def0_u64), &p4);
    let b4 = DynPrime::<4>::new(U256::from(0x0fed_cba9_8765_4321_u64), &p4);
    let t = Instant::now();
    for _ in 0..ITERS {
      a4 *= b4;
    }
    let ns4 = t.elapsed().as_nanos() as f64 / ITERS as f64;
    std::hint::black_box(a4);
    let t = Instant::now();
    for _ in 0..ITERS {
      a2 += b2;
    }
    let add2 = t.elapsed().as_nanos() as f64 / ITERS as f64;
    std::hint::black_box(a2);
    let t = Instant::now();
    for _ in 0..ITERS {
      a4 += b4;
    }
    let add4 = t.elapsed().as_nanos() as f64 / ITERS as f64;
    std::hint::black_box(a4);
    println!(
      "mul: DynPrime<2> {ns2:.2} ns  DynPrime<4> {ns4:.2} ns  ratio {:.2}x",
      ns4 / ns2
    );
    println!(
      "add: DynPrime<2> {add2:.2} ns  DynPrime<4> {add4:.2} ns  ratio {:.2}x",
      add4 / add2
    );
  }

  // Small Mersenne prime 2^61 - 1, easy to verify against u128 arithmetic.
  fn test_params() -> FixedMontyParams<4> {
    let modulus: U256 = U256::from(0x1fff_ffff_ffff_ffff_u64);
    FixedMontyParams::new(Odd::new(modulus).unwrap())
  }

  #[test]
  fn add_mul_reduce_mod_modulus() {
    let p = test_params();
    let p_int: u128 = 0x1fff_ffff_ffff_ffff;

    let a_int: u64 = 0x1234_5678_9abc_def0;
    let b_int: u64 = 0x0fed_cba9_8765_4321;
    let a = DynPrime::<4>::from_u64(&p, a_int);
    let b = DynPrime::<4>::from_u64(&p, b_int);

    let sum_int = ((a_int as u128) + (b_int as u128)) % p_int;
    let sum_low = u64::from_le_bytes((a + b).to_le_bytes()[..8].try_into().unwrap());
    assert_eq!(sum_low as u128, sum_int);

    let prod_int = ((a_int as u128) * (b_int as u128)) % p_int;
    let prod_low = u64::from_le_bytes((a * b).to_le_bytes()[..8].try_into().unwrap());
    assert_eq!(prod_low as u128, prod_int);
  }

  #[test]
  fn neg_then_add_gives_zero() {
    let p = test_params();
    let a = DynPrime::<4>::from_u64(&p, 12345);
    let zero = DynPrime::<4>::zero(&p);
    assert_eq!(a + (-a), zero);
  }

  #[test]
  fn invert_then_mul_gives_one() {
    let p = test_params();
    let a = DynPrime::<4>::from_u64(&p, 7);
    let a_inv = a.invert().unwrap();
    assert_eq!(a * a_inv, DynPrime::<4>::one(&p));
  }

  #[test]
  fn zero_has_no_inverse() {
    let p = test_params();
    let zero = DynPrime::<4>::zero(&p);
    assert!(zero.invert().is_none());
  }

  #[test]
  fn from_bytes_reduce_handles_wide_values() {
    let p = test_params();
    // 64-byte value (512 bits) — wider than Uint<4> (256 bits).
    let mut bytes = vec![0u8; 64];
    bytes[32] = 1; // represents 2^256
    let result = DynPrime::<4>::from_bytes_reduce(&p, &bytes);
    // Verify by computing 2^256 mod p via repeated squaring in DynPrime
    let two = DynPrime::<4>::from_u64(&p, 2);
    let mut pow = DynPrime::<4>::one(&p);
    for _ in 0..256 {
      pow *= two;
    }
    assert_eq!(
      result, pow,
      "from_bytes_reduce should correctly reduce values wider than LIMBS"
    );
  }
}
