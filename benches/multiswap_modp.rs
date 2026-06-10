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
//! Fidelity (see docs/multiswap_modp_bench_plan.md): the arithmetic
//! **core** is real — real RSA-2048 `N`, a real 352-bit `ℓ`, real values
//! < the row modulus, correct quotients — so the Phase-3 D5 range checks
//! run at true ~2048-bit width (numlimb ≈ 64). The hashes (`H`, `Hp`,
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

/// Limb bound (bits) for the IntEval range checks. Smaller limbs keep the
/// Partial-Eval-Norm ceiling on `log P` high at the cost of more limbs
/// (`numlimb = ⌈log_t_f / log_t⌉`); see the bound discussion in
/// docs/multiswap_modp_bench_plan.md.
const LOG_T: usize = 32;

/// Base-hash model: imod rows charged per `H` invocation. The base hash
/// is field arithmetic mod a fixed prime; in the imod metric it does *not*
/// blow up (unlike the paper's F_p limb-split representation). This is a
/// modeling knob, not a faithful Poseidon circuit.
const H_ROWS: usize = 8;

/// Hash-to-prime (`Hp`) / Pocklington model: imod rows for the prime
/// challenge generation, charged once per MultiSwap proof. Modeled at
/// `ℓ`-width (the dominant final Pocklington exponentiation is mod a
/// ~322-bit prime).
const HP_ROWS: usize = 600;

/// Number of group exponentiations per MultiSwap proof (paper Fig. 3:
/// `4·c_eG(|ℓ|)`): ⟦S⟧^e_ins, Q_ins^ℓ, ⟦S'⟧^e_rm, Q_rm^ℓ.
const N_GROUP_EXPS: usize = 4;
/// Group mults per MultiSwap proof (paper Fig. 3: `2·c_×G`).
const N_GROUP_MULS: usize = 2;

/// Per-exponentiation modular-multiply count for a `|ℓ|`-bit exponent via
/// square-and-multiply (`≈ 1.5·|ℓ|`: |ℓ| squarings + ~|ℓ|/2 multiplies).
fn exp_len(ell_bits: usize) -> usize {
  ell_bits + ell_bits / 2
}

/// Structural dimensions of the modeled MultiSwap circuit. Pulled into a
/// struct so the smoke test can shrink every axis while exercising the
/// identical (real 2048-bit) arithmetic path.
#[derive(Clone, Copy)]
struct Dims {
  /// Batch size (number of swaps).
  k: usize,
  /// Modular multiplies per group exponentiation.
  exp_len: usize,
  /// Number of group exponentiations (mod N).
  n_group_exps: usize,
  /// Group mults (mod N).
  n_group_muls: usize,
  /// Hash-to-prime model rows (mod ℓ).
  hp_rows: usize,
  /// Base-hash model rows per invocation (mod p_hash).
  h_rows: usize,
}

impl Dims {
  /// Real MultiSwap profile for batch size `k`, with `|ℓ| = 352`.
  fn multiswap(k: usize) -> Self {
    Self {
      k,
      exp_len: exp_len(352),
      n_group_exps: N_GROUP_EXPS,
      n_group_muls: N_GROUP_MULS,
      hp_rows: HP_ROWS,
      h_rows: H_ROWS,
    }
  }
}

/// RSA-2048 challenge number `N` (paper Appendix B) — the ~2048-bit group
/// modulus for the `mod N` group operations.
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

/// A real 352-bit modulus modeling the Fiat-Shamir prime challenge `ℓ`.
/// The IntMod-R1CS relation `a·b = c + m·q` is valid for any modulus, so
/// `ℓ` need not be prime here — only its ~352-bit width matters for the
/// range-check cost.
fn modulus_ell() -> BigUint {
  // 44 bytes = 352 bits, value 0xC3C3…C3 (odd: low byte 0xC3).
  BigUint::from_bytes_be(&[0xc3u8; 44])
}

/// BLS12-381 scalar field prime (paper Appendix B) — the ~255-bit field
/// modeling the base hash `H`'s arithmetic.
fn modulus_p_hash() -> BigUint {
  let hex = "73eda753299d7d483339d80809a1d80553bda402fffe5bfeffffffff00000001";
  BigUint::parse_bytes(hex.as_bytes(), 16).expect("valid BLS12-381 scalar hex")
}

/// Build the per-row modulus list for the modeled MultiSwap circuit, in a
/// fixed canonical order: group exponentiations (mod N), group mults
/// (mod N), Hp/Pocklington (mod ℓ), per-swap ∏H∆ reductions (mod ℓ), the
/// one ∆-reduction (mod ℓ), then the base-hash blocks (mod p_hash).
fn multiswap_row_moduli(d: Dims) -> Vec<BigUint> {
  let n = modulus_n();
  let ell = modulus_ell();
  let p_hash = modulus_p_hash();

  let mut mods = Vec::new();
  // 4 group exponentiations + 2 group mults, all mod N.
  for _ in 0..(d.n_group_exps * d.exp_len + d.n_group_muls) {
    mods.push(n.clone());
  }
  // Hash-to-prime / Pocklington, mod ℓ.
  for _ in 0..d.hp_rows {
    mods.push(ell.clone());
  }
  // Per-swap ∏ H∆ mod ℓ for insertion + removal, plus one ∆ mod ℓ.
  for _ in 0..(2 * d.k + 1) {
    mods.push(ell.clone());
  }
  // Base hash H per element (insertion + removal), modeled as h_rows
  // modular multiplies mod p_hash each.
  for _ in 0..(2 * d.k * d.h_rows) {
    mods.push(p_hash.clone());
  }
  mods
}

/// Operand scaffolding for the modeled circuit: one `(aᵣ, bᵣ, mᵣ)` triple
/// per real row, with `aᵣ, bᵣ` just below the row modulus `mᵣ` (large,
/// distinct, `< mᵣ`). This is cheap bookkeeping (subtractions only) — the
/// expensive multiprecision work lives in [`compute_advice`].
fn multiswap_operands(d: Dims) -> Vec<(BigUint, BigUint, BigUint)> {
  multiswap_row_moduli(d)
    .into_iter()
    .enumerate()
    .map(|(r, m)| {
      let a = &m - BigUint::from((r as u64 % 17) + 1);
      let b = &m - BigUint::from(((r as u64 * 7) % 19) + 2);
      (a, b, m)
    })
    .collect()
}

/// Multiprecision **advice** generation: per row compute the product
/// `prod = aᵣ·bᵣ` and divide to get `(qᵣ, cᵣ)` with `prod = qᵣ·mᵣ + cᵣ`,
/// `0 ≤ cᵣ < mᵣ`. Returns `(c values, q values)`.
///
/// This is the imod analog of MultiSwap's witness advice (paper §4.3-4.4):
/// the quotient divisions, and — for the `mod ℓ` rows — the `∏ H∆ mod ℓ`
/// product/reduction steps. It is the prover work that *precedes* the
/// SNARK proof and is excluded from the `prove` timing, so the `advice`
/// benchmark measures it on its own. (One big-int multiply + one divmod
/// per row; `~2048×2048→4096`-bit for the `mod N` rows.)
fn compute_advice(operands: &[(BigUint, BigUint, BigUint)]) -> (Vec<BigUint>, Vec<BigUint>) {
  operands
    .iter()
    .map(|(a, b, m)| {
      let prod = a * b;
      let (qi, c) = prod.div_rem(m); // prod = qi·m + c, 0 ≤ c < m
      (c, qi)
    })
    .unzip()
}

/// Build a valid IntMod-R1CS instance for the modeled MultiSwap circuit
/// from the operand scaffolding ([`multiswap_operands`]) and the
/// multiprecision advice ([`compute_advice`]). Each real row `r` is one
/// modular multiply `aᵣ·bᵣ = cᵣ + mᵣ·qᵣ` over Z. Witness columns are laid
/// out `w = [a_0,b_0,c_0, a_1,b_1,c_1, …]` padded to a power of two;
/// `num_cons` is the next power of two ≥ the real row count, padding rows
/// being the trivial `0 = 0`.
fn multiswap_shape_and_witness(d: Dims) -> (IntModR1CSShapeModp<M>, Vec<BigUint>, Vec<BigUint>) {
  let operands = multiswap_operands(d);
  let (cs, qs) = compute_advice(&operands);
  let num_real = operands.len();
  let num_cons = num_real.next_power_of_two();
  let num_vars = (3 * num_real).next_power_of_two();
  let num_io = 0;

  let mut a_entries = Vec::with_capacity(num_real);
  let mut b_entries = Vec::with_capacity(num_real);
  let mut c_entries = Vec::with_capacity(num_real);
  let mut w = vec![BigUint::from(0u32); num_vars];
  let one = BigUint::from(1u32);

  for (r, ((a, b, _m), c)) in operands.iter().zip(cs.iter()).enumerate() {
    let (ca, cb, cc) = (3 * r, 3 * r + 1, 3 * r + 2);
    a_entries.push((r, ca, one.clone()));
    b_entries.push((r, cb, one.clone()));
    c_entries.push((r, cc, one.clone()));
    w[ca] = a.clone();
    w[cb] = b.clone();
    w[cc] = c.clone();
  }

  // q holds one quotient per real row; pad to num_cons (padding rows are
  // the trivial `0 = 0`).
  let mut q = qs;
  q.resize(num_cons, BigUint::from(0u32));

  // Per-row moduli, padded to num_cons (padding rows valid for any modulus).
  let mut mods: Vec<BigUint> = operands.into_iter().map(|(_, _, m)| m).collect();
  mods.resize(num_cons, BigUint::from(2u32));

  let shape = IntModR1CSShapeModp::<M>::new(
    num_cons, num_vars, num_io, a_entries, b_entries, c_entries, mods,
  )
  .expect("valid IntMod-R1CS shape");

  (shape, w, q)
}

/// Derive IntEval params sized for the shape's committed-value width
/// (~2048-bit `mod N` operands → `log_t_f = 2048`) and vector length.
fn params_for(shape: &IntModR1CSShapeModp<M>, int_k: usize) -> IntEvalParams {
  let n = shape.num_vars().max(shape.num_cons());
  let log_n = (n as u64).ilog2() as usize; // n is a power of two
  IntEvalParams::derive(2048, LOG_T, int_k, log_n).expect("IntEval params satisfy bounds")
}

/// Paper Fig. 3 analytical F_p constraint count for MultiSwap at batch
/// size `k`, for the headline imod-vs-xJsnark ratio. Per-op (×2 for
/// insert+remove) plus the fixed per-proof overhead.
fn paper_fp_constraints(k: usize) -> u64 {
  // Parameter values from Fig. 3 (Poseidon hash: c_He = c_Hin ≈ 316).
  let f = 255u64; // field width
  let b_h_delta = 2048u64; // division-intractable hash output bits
  let ell_bits = 352u64; // |ℓ|
  let c_he = 316; // multiset item hash → F (Poseidon)
  let c_hin = 316; // full-input hash per op
  let c_split = 388;
  let c_add_ell = 16 + f; // c_+ℓ(f)
  let c_mul_ell = 479; // c_×ℓ
  let c_e_g = 7044 * ell_bits; // c_eG(|ℓ|)
  let c_x_g = 7563;
  let c_hp = 217703;
  let c_mod_ell = 16 + b_h_delta; // c_mod_ℓ(b_H∆)

  let per_op = 2 * (c_he + c_hin + c_split + c_add_ell + c_mul_ell);
  let per_proof = 4 * c_e_g + 2 * c_x_g + c_hp + c_mod_ell;
  (k as u64) * per_op + per_proof
}

fn multiswap_modp_benches(c: &mut Criterion) {
  // Bench only k = 0: the pure fixed per-proof overhead of the two
  // Wesolowski proofs (4 group exponentiations + 2 group mults mod N, plus
  // the prime hash Hp), with no per-swap rows. This isolates the cost a
  // MultiSwap pays before any swaps are amortized — the constant that sets
  // its break-even point vs. Merkle trees. Lands at num_cons = 2^12.
  let ks: &[usize] = &[0];

  // Per-part timing breakdown, gated on `RUST_LOG` (mirrors
  // imod_spartan_modp.rs): one full setup/prove/verify per config so the
  // D5 range-check spans (at numlimb ≈ 64) are visible, with an `is_sat`
  // correctness gate.
  if std::env::var_os("RUST_LOG").is_some() {
    let _ = tracing_subscriber::fmt()
      .with_target(false)
      .with_env_filter(EnvFilter::from_default_env())
      .try_init();
    for &k in ks {
      let dims = Dims::multiswap(k);
      let (shape, w, q) = multiswap_shape_and_witness(dims);
      // Sweep the IntEval per-iteration parameter `int_k` (distinct from the
      // MultiSwap batch size `k`) to see how it trades off `s`/iterations
      // against prove/verify cost under the prime-counting Soundness-1 bound.
      for int_k in 7..=10usize {
        let params = params_for(&shape, int_k);
        println!(
          "=== IntEval k={int_k}: log_p={} s={} numlimb={} numlimb_var={} (batch k={k}) ===",
          params.log_p, params.s, params.numlimb, params.numlimb_var
        );
        let (pk, vk) =
          IntModSpartanModpSNARK::<M>::setup_with_params(shape.clone(), params).unwrap();
        let (witness, instance) =
          IntModR1CSWitnessModp::<M>::new(&shape, pk.ck(), w.clone(), q.clone(), vec![]).unwrap();
        shape.is_sat(pk.ck(), &instance, &witness).unwrap();
        let proof = IntModSpartanModpSNARK::<M>::prove(&pk, &instance, &witness).unwrap();
        proof.verify(&vk, &instance).unwrap();
      }
    }
  }

  // Report shape sizes + the paper's analytical F_p count alongside.
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

    // Multiprecision advice generation alone (per-row product + divmod →
    // c, q). The imod analog of the paper's witness advice (Fig. 6's
    // "witness computation", minus the accumulator-digest exponentiation
    // we don't model). Excluded from `prove` — measured here on its own.
    g.bench_function(format!("advice/{tag}"), |b| {
      b.iter_batched(
        || multiswap_operands(dims),
        |operands| {
          let _ = compute_advice(&operands);
        },
        BatchSize::LargeInput,
      );
    });

    // Witness commitment alone (blind + commit w/q). This is the portion
    // of `prove` contributed by `IntModR1CSWitnessModp::new`; subtract it
    // from `prove` to isolate proof generation.
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

    g.bench_function(format!("prove/{tag}"), |b| {
      b.iter_batched(
        || {
          let (shape, w, q) = multiswap_shape_and_witness(dims);
          let params = params_for(&shape, DEFAULT_K);
          let (pk, _vk) =
            IntModSpartanModpSNARK::<M>::setup_with_params(shape.clone(), params).unwrap();
          (pk, shape, w, q)
        },
        |(pk, shape, w, q)| {
          // Time the witness commitment together with proof generation,
          // to match the paper's Fig. 6 ("witness computation + proof
          // generation"). `IntModR1CSWitnessModp::new` blinds and commits
          // w/q to the PCS — a real prover cost. NOTE: the paper's
          // witness-computation term is dominated by the RSA
          // accumulator-digest exponentiation (§4.4, ~43 s at 2^20),
          // which this synthetic instance does not build; only the
          // commitment portion of witness work is captured here.
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
