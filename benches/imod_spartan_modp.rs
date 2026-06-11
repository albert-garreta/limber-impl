//! benches/imod_spartan_modp.rs
//! Criterion benchmarks for Phase-2 IntModSpartanModpSNARK {setup, prove,
//! verify} on synthetic Integer-Mod-R1CS instances of varying size.
//!
//! Mirror of `benches/imod_spartan.rs` but parameterized over the Phase-2
//! driver (T256DynPrimeEngine + IntegerModPCS). Each constraint is one
//! independent modular multiplication `a · b = c + N · q`.
//!
//! Run with:
//!   RUSTFLAGS="-C target-cpu=native" cargo bench --bench imod_spartan_modp
#[cfg(feature = "jem")]
use tikv_jemallocator::Jemalloc;
#[cfg(feature = "jem")]
#[global_allocator]
static GLOBAL: Jemalloc = tikv_jemallocator::Jemalloc;

use criterion::{BatchSize, Criterion, criterion_group, criterion_main};
use ff::{Field, PrimeField};
use num_bigint::{BigUint, RandBigInt};
use rand::SeedableRng;
use spartan2::{
  imod_r1cs_modp::{IntModR1CSShapeModp, IntModR1CSWitnessModp},
  imod_spartan_modp::{
    IntModSpartanModpProverKey, IntModSpartanModpSNARK, IntModSpartanModpVerifierKey,
  },
  provider::T256DynPrimeEngine,
  provider::pcs::integer_modpcs::{DEFAULT_K, DEFAULT_LOG_T_F, IntEvalParams},
  provider::pt256::t256,
};
use std::time::Duration;
use tracing_subscriber::EnvFilter;

type M = T256DynPrimeEngine;

/// Limb bound (bits) for the MultiSwap-shaped config; matches the
/// MultiSwap bench's `LOG_T`.
const MSSHAPE_LOG_T: usize = 32;
/// Norm bound (bits) on committed values for the MultiSwap-shaped
/// config: operands are full T256-scalar-field elements (< 2^256).
const MSSHAPE_LOG_T_F: usize = 256;

/// The T256 **base-field** modulus `p` as a `BigUint`, computed from
/// `-Base::ONE`. This is a foreign 256-bit modulus from the proof
/// system's perspective (the native field is the scalar field `q ≠ p`),
/// so a gate `a·b ≡ c (mod p)` is NOT expressible as one native R1CS
/// constraint — it's the realistic workload class (foreign-field /
/// curve-coordinate arithmetic) that the integer Mod-PCS exists for.
fn t256_base_modulus() -> BigUint {
  let p_minus_1 = (-<t256::Base as Field>::ONE).to_repr();
  BigUint::from_bytes_le(p_minus_1.as_ref()) + 1u32
}

/// Setup honoring an optional `IMOD_K` env override for IntEval's
/// per-iteration variable count `k` (default `DEFAULT_K = 7`), so a
/// sweep can compare derived `(log_p, s, t)` trade-offs without code
/// edits.
fn setup_for(
  shape: IntModR1CSShapeModp<M>,
) -> (
  IntModSpartanModpProverKey<M>,
  IntModSpartanModpVerifierKey<M>,
) {
  match std::env::var("IMOD_K").ok().and_then(|s| s.parse().ok()) {
    Some(k) => {
      let n = shape.num_vars().max(shape.num_cons());
      let log_n = (n as u64).ilog2() as usize;
      let params = IntEvalParams::derive_no_limb_split(DEFAULT_LOG_T_F, k, log_n)
        .expect("IMOD_K params satisfy bounds");
      IntModSpartanModpSNARK::<M>::setup_with_params(shape, params).unwrap()
    }
    None => IntModSpartanModpSNARK::<M>::setup(shape).unwrap(),
  }
}

/// MultiSwap-shaped synthetic config: `2730 = ⌊2^13 / 3⌋` random
/// multiplication gates `a·b ≡ c (mod p)` where `p` is the T256
/// **base-field** modulus — a foreign 256-bit modulus the native proof
/// system (scalar field `q ≠ p`) cannot handle in one constraint — and
/// `a, b` are uniform 256-bit values below `p`. Pads to the same
/// `(2^12 cons, 2^13 vars)` shape as MultiSwap k=0 (real RSA-2048 rows
/// pad identically), and exercises the same machinery —
/// `log_t_f = 256` → `numlimb = 8`, `t = 2` IntEval iterations — at
/// 256-bit instead of 2048-bit width. `spartan_synthetic`'s `msshape`
/// config provides the shape-matched *native baseline* (same R1CS
/// dimensions, full-width values, native gates): it does NOT express
/// this statement — natively it would need limb-decomposition gadgets
/// (~10²-10³ constraints per gate) — so the ratio reads as "machinery
/// overhead vs what the same shape costs natively", not same-statement.
fn make_msshape_shape_and_witness() -> (IntModR1CSShapeModp<M>, Vec<BigUint>, Vec<BigUint>) {
  let num_real: usize = 2730;
  let num_cons = num_real.next_power_of_two(); // 2^12
  let num_vars = (3 * num_real).next_power_of_two(); // 2^13
  let n = t256_base_modulus();
  let mut rng = rand::rngs::StdRng::seed_from_u64(0x6d73_7368);

  let one = BigUint::from(1u32);
  let num_io = 0;
  let mut a_entries = Vec::with_capacity(num_real);
  let mut b_entries = Vec::with_capacity(num_real);
  let mut c_entries = Vec::with_capacity(num_real);
  for i in 0..num_real {
    a_entries.push((i, 3 * i, one.clone()));
    b_entries.push((i, 3 * i + 1, one.clone()));
    c_entries.push((i, 3 * i + 2, one.clone()));
  }
  // Padding rows (i ≥ num_real) are all-zero: 0·0 = 0 + q·0.
  let mods = vec![n.clone(); num_cons];

  let shape = IntModR1CSShapeModp::<M>::new(
    num_cons, num_vars, num_io, a_entries, b_entries, c_entries, mods,
  )
  .unwrap();

  let zero = BigUint::from(0u32);
  let mut w = vec![zero.clone(); num_vars];
  let mut q = vec![zero; num_cons];
  for i in 0..num_real {
    let a = rng.gen_biguint_below(&n);
    let b = rng.gen_biguint_below(&n);
    let ab = &a * &b;
    q[i] = &ab / &n;
    w[3 * i] = a;
    w[3 * i + 1] = b;
    w[3 * i + 2] = &ab % &n;
  }

  (shape, w, q)
}

/// Setup for the MultiSwap-shaped config: `log_t_f = 256` (limb-split
/// into 8 limbs of 32 bits), honoring the `IMOD_K` override.
fn setup_msshape(
  shape: IntModR1CSShapeModp<M>,
) -> (
  IntModSpartanModpProverKey<M>,
  IntModSpartanModpVerifierKey<M>,
) {
  let k = std::env::var("IMOD_K")
    .ok()
    .and_then(|s| s.parse().ok())
    .unwrap_or(DEFAULT_K);
  let n = shape.num_vars().max(shape.num_cons());
  let log_n = (n as u64).ilog2() as usize;
  let params = IntEvalParams::derive(MSSHAPE_LOG_T_F, MSSHAPE_LOG_T, k, log_n)
    .expect("msshape params satisfy bounds");
  IntModSpartanModpSNARK::<M>::setup_with_params(shape, params).unwrap()
}

/// Synthetic shape: `num_cons` independent modular multiplications, each
/// using three fresh witness columns. Witness columns are laid out as
/// `w = [a_0, b_0, c_0, a_1, b_1, c_1, …]` padded to `num_vars`.
fn make_shape_and_witness(
  num_cons: usize,
  num_vars: usize,
) -> (IntModR1CSShapeModp<M>, Vec<BigUint>, Vec<BigUint>) {
  assert!(
    3 * num_cons <= num_vars,
    "num_vars must hold 3·num_cons columns"
  );

  let one = BigUint::from(1u32);
  let num_io = 0;

  let mut a_entries = Vec::with_capacity(num_cons);
  let mut b_entries = Vec::with_capacity(num_cons);
  let mut c_entries = Vec::with_capacity(num_cons);
  for i in 0..num_cons {
    a_entries.push((i, 3 * i, one.clone()));
    b_entries.push((i, 3 * i + 1, one.clone()));
    c_entries.push((i, 3 * i + 2, one.clone()));
  }

  let modulus: u64 = 7;
  let mods = vec![BigUint::from(modulus); num_cons];

  let shape = IntModR1CSShapeModp::<M>::new(
    num_cons, num_vars, num_io, a_entries, b_entries, c_entries, mods,
  )
  .unwrap();

  // Witness: pick small a, b so a·b fits in u64; choose c, q so a·b = c + N·q.
  let zero = BigUint::from(0u32);
  let mut w = vec![zero.clone(); num_vars];
  let mut q = vec![zero; num_cons];
  for i in 0..num_cons {
    let a = (i as u64 % 100) + 1;
    let b = ((i as u64 * 7) % 100) + 1;
    let ab = a * b;
    let qi = ab / modulus;
    let ci = ab % modulus;
    w[3 * i] = BigUint::from(a);
    w[3 * i + 1] = BigUint::from(b);
    w[3 * i + 2] = BigUint::from(ci);
    q[i] = BigUint::from(qi);
  }

  (shape, w, q)
}

fn imod_spartan_modp_benches(c: &mut Criterion) {
  // (num_cons, num_vars) — num_vars is the next power-of-two ≥ 4·num_cons.
  // The Mod-PCS open point has length log_2(num_vars); for num_vars > 2^k
  // (default k = 7) this exercises IntEval's partial-eval iteration path.
  let configs: &[(usize, usize)] = &[
    (1usize << 6, 1usize << 8),   // point.len=8 → t=1 IntEval iteration
    (1usize << 8, 1usize << 10),  // point.len=10 → t=1
    (1usize << 10, 1usize << 12), // point.len=12 → t=1
  ];

  // Per-part timing breakdown, gated entirely on `RUST_LOG` so a plain
  // `cargo bench` installs no global subscriber and runs no extra work.
  // With `RUST_LOG=info` we install a fmt subscriber and run one
  // setup/prove/verify per config, printing the section spans
  // (imod_pcs_chain_openings, imod_pcs_rc_ab, …) so you can see where
  // prove/verify time goes without criterion's iteration noise.
  if std::env::var_os("RUST_LOG").is_some() {
    let _ = tracing_subscriber::fmt()
      .with_target(false)
      .with_env_filter(EnvFilter::from_default_env())
      .try_init();
    for &(num_cons, num_vars) in configs {
      let (shape, w, q) = make_shape_and_witness(num_cons, num_vars);
      let (pk, vk) = setup_for(shape.clone());
      let (witness, instance) =
        IntModR1CSWitnessModp::<M>::new(&shape, pk.ck(), w, q, vec![]).unwrap();
      let proof = IntModSpartanModpSNARK::<M>::prove(&pk, &instance, &witness).unwrap();
      proof.verify(&vk, &instance).unwrap();
    }
    {
      let (shape, w, q) = make_msshape_shape_and_witness();
      let (pk, vk) = setup_msshape(shape.clone());
      let (witness, instance) =
        IntModR1CSWitnessModp::<M>::new(&shape, pk.ck(), w, q, vec![]).unwrap();
      let proof = IntModSpartanModpSNARK::<M>::prove(&pk, &instance, &witness).unwrap();
      proof.verify(&vk, &instance).unwrap();
    }
  }

  let mut g = c.benchmark_group("imod_spartan_modp");
  g.sample_size(10);
  g.warm_up_time(Duration::from_millis(100));
  g.measurement_time(Duration::from_secs(20));

  for &(num_cons, num_vars) in configs {
    let tag = format!("c2^{}_v2^{}", num_cons.ilog2(), num_vars.ilog2());

    g.bench_function(format!("setup/{tag}"), |b| {
      b.iter_batched(
        || make_shape_and_witness(num_cons, num_vars).0,
        |shape| {
          let _ = IntModSpartanModpSNARK::<M>::setup(shape).unwrap();
        },
        BatchSize::LargeInput,
      );
    });

    g.bench_function(format!("prove/{tag}"), |b| {
      b.iter_batched(
        || {
          let (shape, w, q) = make_shape_and_witness(num_cons, num_vars);
          let (pk, _vk) = setup_for(shape.clone());
          let (witness, instance) =
            IntModR1CSWitnessModp::<M>::new(&shape, pk.ck(), w, q, vec![]).unwrap();
          (pk, instance, witness)
        },
        |(pk, instance, witness)| {
          let _ = IntModSpartanModpSNARK::<M>::prove(&pk, &instance, &witness).unwrap();
        },
        BatchSize::LargeInput,
      );
    });

    g.bench_function(format!("verify/{tag}"), |b| {
      b.iter_batched(
        || {
          let (shape, w, q) = make_shape_and_witness(num_cons, num_vars);
          let (pk, vk) = setup_for(shape.clone());
          let (witness, instance) =
            IntModR1CSWitnessModp::<M>::new(&shape, pk.ck(), w, q, vec![]).unwrap();
          let proof = IntModSpartanModpSNARK::<M>::prove(&pk, &instance, &witness).unwrap();
          (vk, instance, proof)
        },
        |(vk, instance, proof)| {
          proof.verify(&vk, &instance).unwrap();
        },
        BatchSize::LargeInput,
      );
    });
  }

  // MultiSwap-shaped config: 2^12 cons / 2^13 vars, full-width gates
  // mod the T256 base-field modulus (foreign to the native scalar
  // field). Shape-matched against plain Spartan's
  // `spartan_synthetic/.../msshape` native baseline.
  {
    let tag = "msshape_c2^12_v2^13";
    g.bench_function(format!("prove/{tag}"), |b| {
      b.iter_batched(
        || {
          let (shape, w, q) = make_msshape_shape_and_witness();
          let (pk, _vk) = setup_msshape(shape.clone());
          let (witness, instance) =
            IntModR1CSWitnessModp::<M>::new(&shape, pk.ck(), w, q, vec![]).unwrap();
          (pk, instance, witness)
        },
        |(pk, instance, witness)| {
          let _ = IntModSpartanModpSNARK::<M>::prove(&pk, &instance, &witness).unwrap();
        },
        BatchSize::LargeInput,
      );
    });

    g.bench_function(format!("verify/{tag}"), |b| {
      b.iter_batched(
        || {
          let (shape, w, q) = make_msshape_shape_and_witness();
          let (pk, vk) = setup_msshape(shape.clone());
          let (witness, instance) =
            IntModR1CSWitnessModp::<M>::new(&shape, pk.ck(), w, q, vec![]).unwrap();
          let proof = IntModSpartanModpSNARK::<M>::prove(&pk, &instance, &witness).unwrap();
          (vk, instance, proof)
        },
        |(vk, instance, proof)| {
          proof.verify(&vk, &instance).unwrap();
        },
        BatchSize::LargeInput,
      );
    });
  }
  g.finish();
}

criterion_group!(benches, imod_spartan_modp_benches);
criterion_main!(benches);
