//! Regression probes for the wide-value (>256-bit) reduction bug:
//! `DynPrime::from_bytes_reduce` truncates its input to 32 bytes, so a
//! value `v ≥ 2^256` reduces as `(v mod 2^256) mod p`, not `v mod p`
//! (confirmed 2026-06-11; tracked in docs/imod_followups.md).
//!
//! Consequence: the Phase-2 SNARK's Z_p layer binds the TRUNCATED
//! witness/moduli, and completeness fails for honest wide instances —
//! UNLESS the instance satisfies the row relation as a polynomial
//! identity in the modulus (e.g. `(m−5)(m−7) = 35 + m(m−12)`), which
//! truncation preserves. The MultiSwap bench's synthetic operands have
//! exactly that structure, which is why it roundtrips despite the bug.
//!
//! `wide_modulus_roundtrip` (structured operands) passes today;
//! `wide_modulus_roundtrip_random_operands` is the honest case and is
//! `#[ignore]`d until the chunked wide reduction lands — it must pass
//! once the bug is fixed.

use num_bigint::BigUint;
use spartan2::{
  imod_r1cs_modp::{IntModR1CSShapeModp, IntModR1CSWitnessModp},
  imod_spartan_modp::IntModSpartanModpSNARK,
  provider::T256DynPrimeEngine,
  provider::pcs::integer_modpcs::{DEFAULT_K, IntEvalParams},
};

type M = T256DynPrimeEngine;

#[test]
fn wide_modulus_roundtrip() {
  wide_roundtrip_inner(false)
}

/// Same instance but with operands that are NOT a polynomial identity in
/// `m` — genuinely random wide values, i.e. the honest general case.
/// Currently FAILS with `InvalidSumcheckProof` (the truncation bug);
/// un-ignore when `from_bytes_reduce` gains chunked wide reduction.
#[test]
#[ignore = "KNOWN BUG: from_bytes_reduce truncates >256-bit values (docs/imod_followups.md)"]
fn wide_modulus_roundtrip_random_operands() {
  wide_roundtrip_inner(true)
}

fn wide_roundtrip_inner(random_operands: bool) {
  use num_bigint::RandBigInt;
  use rand::SeedableRng;
  // One real row a·b = c + m·q with a ~2048-bit modulus, padded to 4
  // rows / 16 vars (mirrors the MultiSwap construction at tiny scale).
  let m: BigUint = (BigUint::from(1u32) << 2047) + BigUint::from(977u32); // odd ~2048-bit modulus
  let (a, b) = if random_operands {
    let mut rng = rand::rngs::StdRng::seed_from_u64(42);
    (rng.gen_biguint_below(&m), rng.gen_biguint_below(&m))
  } else {
    (&m - 5u32, &m - 7u32)
  };
  let ab = &a * &b;
  let q = &ab / &m;
  let c = &ab % &m;

  let num_cons = 4;
  let num_vars = 16;
  let one = BigUint::from(1u32);
  let a_entries = vec![(0usize, 0usize, one.clone())];
  let b_entries = vec![(0usize, 1usize, one.clone())];
  let c_entries = vec![(0usize, 2usize, one.clone())];
  let mut mods = vec![m.clone(); 1];
  mods.resize(num_cons, BigUint::from(2u32));

  let shape =
    IntModR1CSShapeModp::<M>::new(num_cons, num_vars, 0, a_entries, b_entries, c_entries, mods)
      .unwrap();

  let mut w = vec![BigUint::from(0u32); num_vars];
  w[0] = a;
  w[1] = b;
  w[2] = c;
  let mut qv = vec![BigUint::from(0u32); num_cons];
  qv[0] = q;

  let log_n = 4; // log2(16)
  let params = IntEvalParams::derive(2048, 32, DEFAULT_K, log_n).expect("params");
  let (pk, vk) = IntModSpartanModpSNARK::<M>::setup_with_params(shape.clone(), params).unwrap();
  let (witness, instance) =
    IntModR1CSWitnessModp::<M>::new(&shape, pk.ck(), w, qv, vec![]).unwrap();
  shape.is_sat(pk.ck(), &instance, &witness).unwrap();
  let proof = IntModSpartanModpSNARK::<M>::prove(&pk, &instance, &witness).unwrap();
  proof.verify(&vk, &instance).unwrap();
}
