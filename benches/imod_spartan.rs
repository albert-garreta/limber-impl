//! benches/imod_spartan.rs
//! Criterion benchmarks for IntModSpartan {setup, prove, verify} on synthetic
//! Integer-Mod-R1CS instances of varying size. Each constraint is one
//! independent modular multiplication `a · b = c + N · q`.
//!
//! Run with: `RUSTFLAGS="-C target-cpu=native" cargo bench --bench imod_spartan`
#[cfg(feature = "jem")]
use tikv_jemallocator::Jemalloc;
#[cfg(feature = "jem")]
#[global_allocator]
static GLOBAL: Jemalloc = tikv_jemallocator::Jemalloc;

use criterion::{BatchSize, Criterion, criterion_group, criterion_main};
use ff::Field;
use spartan2::{
  imod_r1cs::{IntModR1CSShape, IntModR1CSWitness},
  imod_spartan::IntModSpartanSNARK,
  provider::T256HyraxEngine,
  traits::Engine,
};
use std::time::Duration;

type E = T256HyraxEngine;
type Sc = <E as Engine>::Scalar;

/// Synthetic shape: `num_cons` independent modular multiplications, each
/// using three fresh witness columns. Witness columns are laid out as
/// `w = [a_0, b_0, c_0, a_1, b_1, c_1, …]` padded to `num_vars`.
fn make_shape_and_witness(
  num_cons: usize,
  num_vars: usize,
) -> (IntModR1CSShape<E>, Vec<Sc>, Vec<Sc>) {
  assert!(
    3 * num_cons <= num_vars,
    "num_vars must hold 3·num_cons columns"
  );

  let one = Sc::ONE;
  let num_io = 0;

  let mut a_entries = Vec::with_capacity(num_cons);
  let mut b_entries = Vec::with_capacity(num_cons);
  let mut c_entries = Vec::with_capacity(num_cons);
  for i in 0..num_cons {
    a_entries.push((i, 3 * i, one));
    b_entries.push((i, 3 * i + 1, one));
    c_entries.push((i, 3 * i + 2, one));
  }

  let modulus: u64 = 7;
  let n_scalar = Sc::from(modulus);
  let mods = vec![n_scalar; num_cons];

  let shape = IntModR1CSShape::<E>::from_entries(
    num_cons, num_vars, num_io, &a_entries, &b_entries, &c_entries, mods,
  )
  .unwrap();

  // Witness: pick small a, b so a·b fits in u64; choose c, q so a·b = c + N·q.
  let mut w = vec![Sc::ZERO; num_vars];
  let mut q = vec![Sc::ZERO; num_cons];
  for i in 0..num_cons {
    let a = (i as u64 % 100) + 1;
    let b = ((i as u64 * 7) % 100) + 1;
    let ab = a * b;
    let qi = ab / modulus;
    let ci = ab % modulus;
    w[3 * i] = Sc::from(a);
    w[3 * i + 1] = Sc::from(b);
    w[3 * i + 2] = Sc::from(ci);
    q[i] = Sc::from(qi);
  }

  (shape, w, q)
}

fn imod_spartan_benches(c: &mut Criterion) {
  // (num_cons, num_vars) — num_vars = next power-of-two ≥ 4·num_cons.
  // Smallest config matches `benches/imod_spartan_modp.rs` for direct
  // p=q vs p≠q comparison; larger ones are Phase-1's historical sizes.
  let configs: &[(usize, usize)] = &[
    (1usize << 6, 1usize << 8),
    (1usize << 10, 1usize << 12),
    (1usize << 12, 1usize << 14),
    (1usize << 14, 1usize << 16),
  ];

  // Report proof sizes once (outside measurements).
  for &(num_cons, num_vars) in configs {
    let (shape, w, q) = make_shape_and_witness(num_cons, num_vars);
    let (pk, _vk) = IntModSpartanSNARK::<E>::setup(shape.clone()).unwrap();
    let (witness, instance) = IntModR1CSWitness::<E>::new(&shape, pk.ck(), w, q, vec![]).unwrap();
    let proof = IntModSpartanSNARK::<E>::prove(&pk, &instance, &witness).unwrap();
    let proof_bytes = bincode::serialize(&proof).unwrap();
    println!(
      "IntModSpartan num_cons=2^{} num_vars=2^{}: proof_size={} bytes",
      num_cons.ilog2(),
      num_vars.ilog2(),
      proof_bytes.len()
    );
  }

  let mut g = c.benchmark_group("imod_spartan");
  g.sample_size(10);
  g.warm_up_time(Duration::from_millis(100));
  g.measurement_time(Duration::from_secs(10));

  for &(num_cons, num_vars) in configs {
    let tag = format!("c2^{}_v2^{}", num_cons.ilog2(), num_vars.ilog2());

    g.bench_function(format!("setup/{tag}"), |b| {
      b.iter_batched(
        || make_shape_and_witness(num_cons, num_vars).0,
        |shape| {
          let _ = IntModSpartanSNARK::<E>::setup(shape).unwrap();
        },
        BatchSize::LargeInput,
      );
    });

    g.bench_function(format!("prove/{tag}"), |b| {
      b.iter_batched(
        || {
          let (shape, w, q) = make_shape_and_witness(num_cons, num_vars);
          let (pk, _vk) = IntModSpartanSNARK::<E>::setup(shape.clone()).unwrap();
          let (witness, instance) =
            IntModR1CSWitness::<E>::new(&shape, pk.ck(), w, q, vec![]).unwrap();
          (pk, instance, witness)
        },
        |(pk, instance, witness)| {
          let _ = IntModSpartanSNARK::<E>::prove(&pk, &instance, &witness).unwrap();
        },
        BatchSize::LargeInput,
      );
    });

    g.bench_function(format!("verify/{tag}"), |b| {
      b.iter_batched(
        || {
          let (shape, w, q) = make_shape_and_witness(num_cons, num_vars);
          let (pk, vk) = IntModSpartanSNARK::<E>::setup(shape.clone()).unwrap();
          let (witness, instance) =
            IntModR1CSWitness::<E>::new(&shape, pk.ck(), w, q, vec![]).unwrap();
          let proof = IntModSpartanSNARK::<E>::prove(&pk, &instance, &witness).unwrap();
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

criterion_group!(benches, imod_spartan_benches);
criterion_main!(benches);
