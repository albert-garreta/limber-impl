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
use num_bigint::BigUint;
use spartan2::{
  imod_r1cs_modp::{IntModR1CSShapeModp, IntModR1CSWitnessModp},
  imod_spartan_modp::IntModSpartanModpSNARK,
  provider::T256DynPrimeEngine,
};
use std::time::Duration;

type M = T256DynPrimeEngine;

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
          let (pk, _vk) = IntModSpartanModpSNARK::<M>::setup(shape.clone()).unwrap();
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
          let (pk, vk) = IntModSpartanModpSNARK::<M>::setup(shape.clone()).unwrap();
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
