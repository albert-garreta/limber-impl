//! benches/multiswap_modp.rs
//! Prover/verifier cost of proving **MultiSwap** (Ozdemir, Wahby,
//! Whitehat, Boneh, *Scaling Verifiable Computation Using Efficient Set
//! Accumulators*, USENIX Security 2020, §3–4) with
//! `IntModSpartanModpSNARK`.
//!
//! MultiSwap verifies a batch of `k` swaps against an RSA accumulator by
//! checking two Wesolowski proofs (one batch insertion, one batch
//! removal) that share a Fiat-Shamir prime challenge `ℓ`:
//!   Q_ins^ℓ · ⟦S⟧^(∏ H∆(yᵢ) mod ℓ) = ⟦S'⟧   in  G = (Z/N)*/{±1}
//! and symmetrically for removal. Its cost (paper Fig. 3) is dominated by
//! multiprecision modular arithmetic: 4 group exponentiations with
//! `|ℓ|≈352`-bit exponents mod a `b_N≈2048`-bit modulus `N`, 2 group
//! mults, the hash-to-prime `Hp`, and per-swap `∏ H∆ mod ℓ`.
//!
//! The IntMod-R1CS relation `A·z ∘ B·z = C·z + m∘q` over Z has one
//! **per-row modulus** `mᵢ` and a prover quotient `qᵢ` — so one row is one
//! modular multiply `LC_A·LC_B ≡ LC_C (mod mᵢ)`. A `mod N` multiply that
//! costs ~7044·(2048/352) R1CS constraints in the paper's xJsnark/F_p
//! representation is a single imod row here. This bench measures exactly
//! that collapse.
//!
//! Fidelity (see docs/multiswap_modp_bench_plan.md): the 4 group
//! exponentiations are **real wired square-and-multiply chains** with
//! witness exponents, bit decomposition, and reconstruction constraints.
//! The bases are fixed constants baked into matrix coefficients (avoiding
//! a degree-3 conditional-multiply decomposition). The hashes (`H`, `Hp`,
//! `H∆`) and RSA group structure are *modeled by operation count*, not
//! faithful crypto circuits, and are flagged as such.
//!
//! Run with:
//!   RUSTFLAGS="-C target-cpu=native" cargo bench --bench multiswap_modp
#[cfg(feature = "jem")]
use tikv_jemallocator::Jemalloc;
#[cfg(feature = "jem")]
#[global_allocator]
static GLOBAL: Jemalloc = tikv_jemallocator::Jemalloc;

use criterion::{BatchSize, Criterion, criterion_group, criterion_main};
use num_bigint::BigUint;
use num_integer::Integer;
use spartan2::{
  imod_r1cs_modp::{IntModR1CSShapeModp, IntModR1CSWitnessModp},
  imod_spartan_modp::IntModSpartanModpSNARK,
  provider::{
    T256DynPrimeEngine,
    pcs::integer_modpcs::{DEFAULT_K, IntEvalParams},
  },
};
use std::time::Duration;
use tracing_subscriber::EnvFilter;

type M = T256DynPrimeEngine;

/// Limb bound (bits) for the IntEval range checks.
const LOG_T: usize = 32;

/// Base-hash model: imod rows charged per `H` invocation.
const H_ROWS: usize = 8;

/// Hash-to-prime (`Hp`) / Pocklington model rows.
const HP_ROWS: usize = 600;

/// Number of group exponentiations per MultiSwap proof.
const N_GROUP_EXPS: usize = 4;
/// Group mults per MultiSwap proof.
const N_GROUP_MULS: usize = 2;

/// Exponent bit-length for the Fiat-Shamir prime challenge `ℓ`.
const ELL_BITS: usize = 352;

#[derive(Clone, Copy)]
struct Dims {
  k: usize,
  ell_bits: usize,
  n_group_exps: usize,
  n_group_muls: usize,
  hp_rows: usize,
  h_rows: usize,
}

impl Dims {
  fn multiswap(k: usize) -> Self {
    Self {
      k,
      ell_bits: ELL_BITS,
      n_group_exps: N_GROUP_EXPS,
      n_group_muls: N_GROUP_MULS,
      hp_rows: HP_ROWS,
      h_rows: H_ROWS,
    }
  }

  fn rows_per_exp(&self) -> usize {
    3 * self.ell_bits + 1
  }

  fn cols_per_exp(&self) -> usize {
    3 * self.ell_bits + 1
  }

  fn non_exp_rows(&self) -> usize {
    self.n_group_muls + self.hp_rows + (2 * self.k + 1) + 2 * self.k * self.h_rows
  }

  fn num_real_rows(&self) -> usize {
    self.n_group_exps * self.rows_per_exp() + self.non_exp_rows()
  }

  fn num_real_cols(&self) -> usize {
    self.n_group_exps * self.cols_per_exp() + 3 * self.non_exp_rows()
  }
}

fn modulus_n() -> BigUint {
  let hex = "c7970ceedcc3b0754490201a7aa613cd73911081c790f5f1a8726f463550bb5b\
             7ff0db8e1ea1189ec72f93d1650011bd721aeeacc2acde32a04107f0648c2813\
             a31f5b0b7765ff8b44b4b6ffc93384b646eb09c7cf5e8592d40ea33c80039f35\
             b4f14a04b51f7bfd781be4d1673164ba8eb991c2c4d730bbbe35f592bdef524a\
             f7e8daefd26c66fc02c479af89d64d373f442709439de66ceb955f3ea37d5159\
             f6135809f85334b5cb1813addc80cd05609f10ac6a95ad65872c909525bdad32\
             bc729592642920f24c61dc5b3c3b7923e56b16a4d9d373d8721f24a3fc0f1b31\
             31f55615172866bccc30f95054c824e733a5eb6817f7bc16399d48c6361cc7e5";
  BigUint::parse_bytes(hex.as_bytes(), 16).expect("valid RSA-2048 hex")
}

fn modulus_ell() -> BigUint {
  BigUint::from_bytes_be(&[0xc3u8; 44])
}

fn modulus_p_hash() -> BigUint {
  let hex = "73eda753299d7d483339d80809a1d80553bda402fffe5bfeffffffff00000001";
  BigUint::parse_bytes(hex.as_bytes(), 16).expect("valid BLS12-381 scalar hex")
}

fn exp_bases() -> [BigUint; 4] {
  let n = modulus_n();
  core::array::from_fn(|i| &n - BigUint::from(37u64 * i as u64 + 3))
}

fn exp_exponents(ell_bits: usize) -> [BigUint; 4] {
  core::array::from_fn(|i| {
    let seed = (i as u64 + 1) * 0x0123_4567_89AB_CDEFu64;
    let mut bytes = vec![0u8; ell_bits.div_ceil(8)];
    for (k, b) in bytes.iter_mut().enumerate() {
      *b = ((seed.wrapping_mul(k as u64 + 1).wrapping_add(0xDEAD)) & 0xFF) as u8;
    }
    if !ell_bits.is_multiple_of(8) {
      bytes[0] &= (1u8 << (ell_bits % 8)) - 1;
    }
    let msb_byte = (ell_bits - 1) / 8;
    let msb_bit = (ell_bits - 1) % 8;
    let msb_idx = bytes.len() - 1 - msb_byte;
    bytes[msb_idx] |= 1u8 << msb_bit;
    BigUint::from_bytes_be(&bytes)
  })
}

#[allow(clippy::too_many_arguments)]
fn build_exp_circuit(
  base: &BigUint,
  exponent: &BigUint,
  n: &BigUint,
  ell_bits: usize,
  row_base: usize,
  col_base: usize,
  const_col: usize,
  a_entries: &mut Vec<(usize, usize, BigUint)>,
  b_entries: &mut Vec<(usize, usize, BigUint)>,
  c_entries: &mut Vec<(usize, usize, BigUint)>,
  mods: &mut Vec<BigUint>,
  w: &mut [BigUint],
  q: &mut [BigUint],
) -> usize {
  let one = BigUint::from(1u32);
  let g_minus_1 = base - &one;

  let bit_col = |j: usize| col_base + j;
  let exp_col = col_base + ell_bits;
  let acc_col = |j: usize| col_base + ell_bits + 1 + j;
  let sq_col = |j: usize| col_base + 2 * ell_bits + 1 + j;

  let bits: Vec<u8> = (0..ell_bits)
    .map(|j| {
      let bit_pos = ell_bits - 1 - j;
      u8::from(exponent.bit(bit_pos as u64))
    })
    .collect();

  for j in 0..ell_bits {
    w[bit_col(j)] = BigUint::from(bits[j]);
  }
  w[exp_col] = exponent.clone();

  let mut row = row_base;
  for j in 0..ell_bits {
    let acc_val = if j == 0 {
      one.clone()
    } else {
      w[acc_col(j - 1)].clone()
    };

    // Square row
    let sq_prod = &acc_val * &acc_val;
    let (sq_q, sq_val) = sq_prod.div_rem(n);
    w[sq_col(j)] = sq_val.clone();
    q[row] = sq_q;

    let acc_j_col = if j == 0 { const_col } else { acc_col(j - 1) };
    a_entries.push((row, acc_j_col, one.clone()));
    b_entries.push((row, acc_j_col, one.clone()));
    c_entries.push((row, sq_col(j), one.clone()));
    mods.push(n.clone());
    row += 1;

    // Conditional-multiply row
    let b_val = BigUint::from(bits[j]) * &g_minus_1 + &one;
    let cm_prod = &sq_val * &b_val;
    let (cm_q, acc_next) = cm_prod.div_rem(n);
    w[acc_col(j)] = acc_next;
    q[row] = cm_q;

    a_entries.push((row, sq_col(j), one.clone()));
    b_entries.push((row, bit_col(j), g_minus_1.clone()));
    b_entries.push((row, const_col, one.clone()));
    c_entries.push((row, acc_col(j), one.clone()));
    mods.push(n.clone());
    row += 1;
  }

  // Binary constraints, as EXACT integer rows (modulus 0 ⇒ the m·q term
  // vanishes, so the row enforces b² = b over ℤ, i.e. b ∈ {0,1},
  // unconditionally). Modulus N is also computationally sound here
  // (non-binary solutions within the range bound are benign lifts of
  // 0/1 or nontrivial idempotents of Z_N, and exhibiting the latter
  // factors N) — but mod-0 is assumption-free, costs the same, and stays
  // sound if the pattern is reused for moduli with known factorization
  // (e.g. mod-ℓ exponent bits in a future Hp gadget).
  for j in 0..ell_bits {
    a_entries.push((row, bit_col(j), one.clone()));
    b_entries.push((row, bit_col(j), one.clone()));
    c_entries.push((row, bit_col(j), one.clone()));
    q[row] = BigUint::from(0u32);
    mods.push(BigUint::from(0u32));
    row += 1;
  }

  // Reconstruction
  for j in 0..ell_bits {
    let power = BigUint::from(1u32) << (ell_bits - 1 - j);
    a_entries.push((row, bit_col(j), power));
  }
  b_entries.push((row, const_col, one.clone()));
  c_entries.push((row, exp_col, one.clone()));
  q[row] = BigUint::from(0u32);
  mods.push(n.clone());
  row += 1;

  let expected = base.modpow(exponent, n);
  assert_eq!(
    w[acc_col(ell_bits - 1)],
    expected,
    "exponentiation circuit witness mismatch"
  );

  row - row_base
}

#[allow(clippy::too_many_arguments)]
fn compute_witness_advice(
  bases: &[BigUint; 4],
  exponents: &[BigUint; 4],
  n: &BigUint,
  ell: &BigUint,
  p_hash: &BigUint,
  d: Dims,
) -> Vec<BigUint> {
  let one = BigUint::from(1u32);
  let mut out = Vec::new();

  for i in 0..d.n_group_exps {
    let g_minus_1 = &bases[i] - &one;
    let mut acc = one.clone();
    for j in 0..d.ell_bits {
      let bit_pos = d.ell_bits - 1 - j;
      let bit = u8::from(exponents[i].bit(bit_pos as u64));
      let sq = (&acc * &acc).div_rem(n).1;
      let b_val = BigUint::from(bit) * &g_minus_1 + &one;
      acc = (&sq * &b_val).div_rem(n).1;
    }
    out.push(acc);
  }

  let groups: &[(&BigUint, usize)] = &[
    (n, d.n_group_muls),
    (ell, d.hp_rows),
    (ell, 2 * d.k + 1),
    (p_hash, 2 * d.k * d.h_rows),
  ];
  let mut r = 0usize;
  for &(m, count) in groups {
    for _ in 0..count {
      let a = m - BigUint::from((r as u64 % 17) + 1);
      let b = m - BigUint::from(((r as u64 * 7) % 19) + 2);
      out.push((&a * &b).div_rem(m).1);
      r += 1;
    }
  }

  out
}

fn multiswap_shape_and_witness(d: Dims) -> (IntModR1CSShapeModp<M>, Vec<BigUint>, Vec<BigUint>) {
  let n = modulus_n();
  let ell = modulus_ell();
  let p_hash = modulus_p_hash();
  let bases = exp_bases();
  let exponents = exp_exponents(d.ell_bits);

  let num_cons = d.num_real_rows().next_power_of_two();
  let num_vars = d.num_real_cols().next_power_of_two();
  let num_io = 0;
  let const_col = num_vars;

  let mut a_entries = Vec::new();
  let mut b_entries = Vec::new();
  let mut c_entries = Vec::new();
  let mut mods = Vec::new();
  let mut w = vec![BigUint::from(0u32); num_vars];
  let mut q = vec![BigUint::from(0u32); num_cons];
  let one = BigUint::from(1u32);

  for i in 0..d.n_group_exps {
    build_exp_circuit(
      &bases[i],
      &exponents[i],
      &n,
      d.ell_bits,
      i * d.rows_per_exp(),
      i * d.cols_per_exp(),
      const_col,
      &mut a_entries,
      &mut b_entries,
      &mut c_entries,
      &mut mods,
      &mut w,
      &mut q,
    );
  }

  let mut row = d.n_group_exps * d.rows_per_exp();
  let mut col = d.n_group_exps * d.cols_per_exp();

  let groups: Vec<(&BigUint, usize)> = vec![
    (&n, d.n_group_muls),
    (&ell, d.hp_rows),
    (&ell, 2 * d.k + 1),
    (&p_hash, 2 * d.k * d.h_rows),
  ];
  let mut r = 0usize;
  for (m, count) in &groups {
    for _ in 0..*count {
      let a_val = *m - BigUint::from((r as u64 % 17) + 1);
      let b_val = *m - BigUint::from(((r as u64 * 7) % 19) + 2);
      let prod = &a_val * &b_val;
      let (qi, ci) = prod.div_rem(m);
      w[col] = a_val;
      w[col + 1] = b_val;
      w[col + 2] = ci;
      q[row] = qi;
      a_entries.push((row, col, one.clone()));
      b_entries.push((row, col + 1, one.clone()));
      c_entries.push((row, col + 2, one.clone()));
      mods.push((*m).clone());
      row += 1;
      col += 3;
      r += 1;
    }
  }

  mods.resize(num_cons, BigUint::from(2u32));

  let shape = IntModR1CSShapeModp::<M>::new(
    num_cons, num_vars, num_io, a_entries, b_entries, c_entries, mods,
  )
  .expect("valid IntMod-R1CS shape");

  (shape, w, q)
}

fn params_for(shape: &IntModR1CSShapeModp<M>, int_k: usize) -> IntEvalParams {
  let n = shape.num_vars().max(shape.num_cons());
  let log_n = (n as u64).ilog2() as usize;
  IntEvalParams::derive(2048, LOG_T, int_k, log_n).expect("IntEval params satisfy bounds")
}

fn paper_fp_constraints(k: usize) -> u64 {
  let f = 255u64;
  let b_h_delta = 2048u64;
  let ell_bits = 352u64;
  let c_he = 316;
  let c_hin = 316;
  let c_split = 388;
  let c_add_ell = 16 + f;
  let c_mul_ell = 479;
  let c_e_g = 7044 * ell_bits;
  let c_x_g = 7563;
  let c_hp = 217703;
  let c_mod_ell = 16 + b_h_delta;

  let per_op = 2 * (c_he + c_hin + c_split + c_add_ell + c_mul_ell);
  let per_proof = 4 * c_e_g + 2 * c_x_g + c_hp + c_mod_ell;
  (k as u64) * per_op + per_proof
}

fn multiswap_modp_benches(c: &mut Criterion) {
  let ks: &[usize] = &[0];

  if std::env::var_os("RUST_LOG").is_some() {
    let _ = tracing_subscriber::fmt()
      .with_target(false)
      .with_env_filter(EnvFilter::from_default_env())
      .try_init();
    for &k in ks {
      let dims = Dims::multiswap(k);
      let (shape, w, q) = multiswap_shape_and_witness(dims);
      for int_k in 7..=10usize {
        let params = params_for(&shape, int_k);
        println!(
          "=== IntEval k={int_k}: log_p={} s={} numlimb={} numlimb_var={} (batch k={k}, cons=2^{}, vars=2^{}) ===",
          params.log_p,
          params.s,
          params.numlimb,
          params.numlimb_var,
          (shape.num_cons() as u64).ilog2(),
          (shape.num_vars() as u64).ilog2(),
        );
        let (pk, vk) =
          IntModSpartanModpSNARK::<M>::setup_with_params(shape.clone(), params).unwrap();
        let (witness, instance) =
          IntModR1CSWitnessModp::<M>::new(&shape, pk.ck(), w.clone(), q.clone(), vec![]).unwrap();
        shape.is_sat(pk.ck(), &instance, &witness).unwrap();
        println!("is_sat passed");
        let proof = IntModSpartanModpSNARK::<M>::prove(&pk, &instance, &witness).unwrap();
        println!("prove passed");
        proof.verify(&vk, &instance).unwrap();
        println!("verify passed");
      }
    }
  }

  for &k in ks {
    let dims = Dims::multiswap(k);
    let (shape, _w, _q) = multiswap_shape_and_witness(dims);
    println!(
      "MultiSwap k={k}: num_cons=2^{} num_vars=2^{} (imod rows) vs paper F_p≈{} constraints",
      (shape.num_cons() as u64).ilog2(),
      (shape.num_vars() as u64).ilog2(),
      paper_fp_constraints(k),
    );
  }

  let mut g = c.benchmark_group("multiswap_modp");
  g.sample_size(10);
  g.warm_up_time(Duration::from_millis(100));
  g.measurement_time(Duration::from_secs(20));

  for &k in ks {
    let dims = Dims::multiswap(k);
    let (shape0, _, _) = multiswap_shape_and_witness(dims);
    let tag = format!("k{k}_c2^{}", (shape0.num_cons() as u64).ilog2());

    g.bench_function(format!("setup/{tag}"), |b| {
      b.iter_batched(
        || {
          let (shape, _, _) = multiswap_shape_and_witness(dims);
          let params = params_for(&shape, DEFAULT_K);
          (shape, params)
        },
        |(shape, params)| {
          let _ = IntModSpartanModpSNARK::<M>::setup_with_params(shape, params).unwrap();
        },
        BatchSize::LargeInput,
      );
    });

    g.bench_function(format!("advice/{tag}"), |b| {
      b.iter_batched(
        || {
          (
            exp_bases(),
            exp_exponents(dims.ell_bits),
            modulus_n(),
            modulus_ell(),
            modulus_p_hash(),
          )
        },
        |(bases, exponents, n, ell, p_hash)| {
          let _ = compute_witness_advice(&bases, &exponents, &n, &ell, &p_hash, dims);
        },
        BatchSize::LargeInput,
      );
    });

    g.bench_function(format!("commit_witness/{tag}"), |b| {
      b.iter_batched(
        || {
          let (shape, w, q) = multiswap_shape_and_witness(dims);
          let params = params_for(&shape, DEFAULT_K);
          let (pk, _vk) =
            IntModSpartanModpSNARK::<M>::setup_with_params(shape.clone(), params).unwrap();
          (pk, shape, w, q)
        },
        |(pk, shape, w, q)| {
          let _ = IntModR1CSWitnessModp::<M>::new(&shape, pk.ck(), w, q, vec![]).unwrap();
        },
        BatchSize::LargeInput,
      );
    });

    // Timed region = the full prover pipeline: witness generation
    // (`multiswap_shape_and_witness`, dominated by the real RSA-2048
    // exponentiation advice) + witness commitment + prove. The untimed
    // setup closure holds only the SNARK setup (PCS key derivation); the
    // shape it builds there is discarded except for the keys, and is
    // regenerated alongside the witness in the routine. `witness_advice`
    // and `commit_witness` above isolate the two pre-prove phases.
    g.bench_function(format!("prove/{tag}"), |b| {
      b.iter_batched(
        || {
          let (shape, _, _) = multiswap_shape_and_witness(dims);
          let params = params_for(&shape, DEFAULT_K);
          let (pk, _vk) = IntModSpartanModpSNARK::<M>::setup_with_params(shape, params).unwrap();
          pk
        },
        |pk| {
          let (shape, w, q) = multiswap_shape_and_witness(dims);
          let (witness, instance) =
            IntModR1CSWitnessModp::<M>::new(&shape, pk.ck(), w, q, vec![]).unwrap();
          let _ = IntModSpartanModpSNARK::<M>::prove(&pk, &instance, &witness).unwrap();
        },
        BatchSize::LargeInput,
      );
    });

    g.bench_function(format!("verify/{tag}"), |b| {
      b.iter_batched(
        || {
          let (shape, w, q) = multiswap_shape_and_witness(dims);
          let params = params_for(&shape, DEFAULT_K);
          let (pk, vk) =
            IntModSpartanModpSNARK::<M>::setup_with_params(shape.clone(), params).unwrap();
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

criterion_group!(benches, multiswap_modp_benches);
criterion_main!(benches);
