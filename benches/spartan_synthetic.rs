//! benches/spartan_synthetic.rs
//! Plain-Spartan baseline for direct comparison with `imod_spartan.rs`.
//! The circuit is N independent multiplications `a_i · b_i = c_i`, picked
//! so that the resulting R1CS shape sizes line up with the imod bench
//! (`num_cons` ≈ N, `num_vars` ≈ 3N rounded up to a power of two).
//!
//! Run with: `RUSTFLAGS="-C target-cpu=native" cargo bench --bench spartan_synthetic`
#[cfg(feature = "jem")]
use tikv_jemallocator::Jemalloc;
#[cfg(feature = "jem")]
#[global_allocator]
static GLOBAL: Jemalloc = tikv_jemallocator::Jemalloc;

use bellpepper_core::{ConstraintSystem, SynthesisError, num::AllocatedNum};
use criterion::{BatchSize, Criterion, criterion_group, criterion_main};
use ff::PrimeField;
use spartan2::{
  provider::T256HyraxEngine,
  spartan::SpartanSNARK,
  traits::{Engine, circuit::SpartanCircuit, snark::R1CSSNARKTrait},
};
use std::{marker::PhantomData, time::Duration};

type E = T256HyraxEngine;

#[derive(Clone, Debug)]
struct SyntheticMulCircuit<F: PrimeField> {
  n: usize,
  _p: PhantomData<F>,
}

impl<F: PrimeField> SyntheticMulCircuit<F> {
  fn new(n: usize) -> Self {
    Self { n, _p: PhantomData }
  }
}

impl<E: Engine> SpartanCircuit<E> for SyntheticMulCircuit<E::Scalar> {
  fn public_values(&self) -> Result<Vec<E::Scalar>, SynthesisError> {
    Ok(vec![])
  }

  fn shared<CS: ConstraintSystem<E::Scalar>>(
    &self,
    _: &mut CS,
  ) -> Result<Vec<AllocatedNum<E::Scalar>>, SynthesisError> {
    Ok(vec![])
  }

  fn precommitted<CS: ConstraintSystem<E::Scalar>>(
    &self,
    _: &mut CS,
    _: &[AllocatedNum<E::Scalar>],
  ) -> Result<Vec<AllocatedNum<E::Scalar>>, SynthesisError> {
    Ok(vec![])
  }

  fn num_challenges(&self) -> usize {
    0
  }

  fn synthesize<CS: ConstraintSystem<E::Scalar>>(
    &self,
    cs: &mut CS,
    _: &[AllocatedNum<E::Scalar>],
    _: &[AllocatedNum<E::Scalar>],
    _: Option<&[E::Scalar]>,
  ) -> Result<(), SynthesisError> {
    for i in 0..self.n {
      let a = AllocatedNum::alloc(cs.namespace(|| format!("a_{i}")), || {
        Ok(E::Scalar::from((i as u64 % 100) + 1))
      })?;
      let b = AllocatedNum::alloc(cs.namespace(|| format!("b_{i}")), || {
        Ok(E::Scalar::from(((i as u64 * 7) % 100) + 1))
      })?;
      let _c = a.mul(cs.namespace(|| format!("c_{i}")), &b)?;
    }
    Ok(())
  }
}

fn spartan_synthetic_benches(c: &mut Criterion) {
  // Match imod_spartan(_modp) bench: num_cons targets 2^k for k ∈ {6, 8,
  // 10, 12, 14}. Plain Spartan pads internally, so the realised shape
  // may be slightly larger than N.
  let configs: &[usize] = &[1 << 6, 1 << 8, 1 << 10, 1 << 12, 1 << 14];

  for &n in configs {
    let circuit = SyntheticMulCircuit::<<E as Engine>::Scalar>::new(n);
    let (pk, _vk) = SpartanSNARK::<E>::setup(circuit.clone()).unwrap();
    let prep = SpartanSNARK::<E>::prep_prove(&pk, circuit.clone(), true).unwrap();
    let (proof, _) = SpartanSNARK::<E>::prove(&pk, circuit, prep, true).unwrap();
    let proof_bytes = bincode::serialize(&proof).unwrap();
    println!(
      "PlainSpartan synthetic n=2^{}: proof_size={} bytes",
      (n as u64).ilog2(),
      proof_bytes.len()
    );
  }

  let mut g = c.benchmark_group("spartan_synthetic");
  g.sample_size(10);
  g.warm_up_time(Duration::from_millis(100));
  g.measurement_time(Duration::from_secs(10));

  for &n in configs {
    let tag = format!("n2^{}", (n as u64).ilog2());

    g.bench_function(format!("setup/{tag}"), |b| {
      b.iter(|| {
        let circuit = SyntheticMulCircuit::<<E as Engine>::Scalar>::new(n);
        let _ = SpartanSNARK::<E>::setup(circuit).unwrap();
      });
    });

    g.bench_function(format!("prep_prove/{tag}"), |b| {
      b.iter_batched(
        || {
          let circuit = SyntheticMulCircuit::<<E as Engine>::Scalar>::new(n);
          SpartanSNARK::<E>::setup(circuit).unwrap().0
        },
        |pk| {
          let circuit = SyntheticMulCircuit::<<E as Engine>::Scalar>::new(n);
          let _ = SpartanSNARK::<E>::prep_prove(&pk, circuit, true).unwrap();
        },
        BatchSize::LargeInput,
      );
    });

    g.bench_function(format!("prove/{tag}"), |b| {
      b.iter_batched(
        || {
          let circuit = SyntheticMulCircuit::<<E as Engine>::Scalar>::new(n);
          let (pk, _vk) = SpartanSNARK::<E>::setup(circuit.clone()).unwrap();
          let prep = SpartanSNARK::<E>::prep_prove(&pk, circuit.clone(), true).unwrap();
          let (_proof, prep_back) =
            SpartanSNARK::<E>::prove(&pk, circuit.clone(), prep, true).unwrap();
          (pk, circuit, prep_back)
        },
        |(pk, circuit, prep)| {
          let _ = SpartanSNARK::<E>::prove(&pk, circuit, prep, true).unwrap();
        },
        BatchSize::LargeInput,
      );
    });

    g.bench_function(format!("verify/{tag}"), |b| {
      b.iter_batched(
        || {
          let circuit = SyntheticMulCircuit::<<E as Engine>::Scalar>::new(n);
          let (pk, vk) = SpartanSNARK::<E>::setup(circuit.clone()).unwrap();
          let prep = SpartanSNARK::<E>::prep_prove(&pk, circuit.clone(), true).unwrap();
          let (proof, _) = SpartanSNARK::<E>::prove(&pk, circuit, prep, true).unwrap();
          (vk, proof)
        },
        |(vk, proof)| {
          proof.verify(&vk).unwrap();
        },
        BatchSize::LargeInput,
      );
    });
  }
  g.finish();
}

criterion_group!(benches, spartan_synthetic_benches);
criterion_main!(benches);
