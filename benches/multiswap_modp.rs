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
//! Fidelity: the 4 group
//! exponentiations are **real wired square-and-multiply chains** with
//! witness exponents, bit decomposition, and reconstruction constraints.
//! The bases are fixed constants baked into matrix coefficients (avoiding
//! a degree-3 conditional-multiply decomposition). The hashes (`H`, `Hp`,
//! `H∆`) and RSA group structure are *modeled by operation count*, not
//! faithful crypto circuits, and are flagged as such. As of the
//! faithful-cost extension, `Hp` is charged at faithful cost and
//! structure: 600 Pocklington-exponentiation rows + 3 chained
//! Poseidon-cost permutations (243 mul rows each, synthetic operands,
//! real chain shape and modulus) + 640 decomposition bit rows + 10
//! reconstruction rows — ~2.0k rows total vs the paper's 217,703 F_p
//! constraints for the same component.
//!
//! Run with:
//!   RUSTFLAGS="-C target-cpu=native" cargo bench --bench multiswap_modp
#[cfg(feature = "jem")]
use tikv_jemallocator::Jemalloc;
#[cfg(feature = "jem")]
#[global_allocator]
static GLOBAL: Jemalloc = tikv_jemallocator::Jemalloc;

use criterion::{BatchSize, Criterion, criterion_group, criterion_main};
use limber::{
  imod_r1cs_modp::{IntModR1CSShapeModp, IntModR1CSWitnessModp},
  imod_spartan_modp::IntModSpartanModpSNARK,
  provider::{
    T256DynPrimeEngine,
    pcs::integer_modpcs::{DEFAULT_K, IntEvalParams},
  },
};
use num_bigint::BigUint;
use num_integer::Integer;
use std::time::Duration;
use tracing_subscriber::EnvFilter;

type M = T256DynPrimeEngine;

/// Limb bound (bits) for the IntEval range checks.
const LOG_T: usize = 64;

/// Base-hash model: imod rows charged per `H` invocation.
const H_ROWS: usize = 8;

/// Hash-to-prime (`Hp`) Pocklington certificate: number of wired
/// square-and-multiply chains and exponent bits per chain. 4 chains of
/// 50 bits — 4·(3·50+1) = 604 rows — matching the ~600-row operation
/// count of the earlier model, but as REAL wired chains (bit
/// decomposition, reconstruction, chained accumulators) over the
/// Mersenne-prime moduli 2^61−1, 2^89−1, 2^107−1, 2^127−1.
const HP_EXPS: usize = 4;
const HP_EXP_BITS: usize = 50;

/// Faithful-cost Poseidon permutation: 81 x^5 S-boxes × 3 mul rows
/// (x², x⁴, x⁵); the MDS and round-constant layers are linear and fold
/// into the LCs for free. Operands are synthetic but the operation
/// count, chaining structure, and modulus are faithful.
const POSEIDON_ROWS_PER_PERM: usize = 243;
/// Poseidon permutations charged inside one `Hp` invocation
/// (candidate generation for the Pocklington chain).
const HP_POSEIDON_PERMS: usize = 3;
/// Bit rows for the Pocklington side-condition decompositions: the
/// Poseidon output (255 bits) and the four chain outputs (61 + 89 +
/// 107 + 127 bits) are fully bit-decomposed — 639 exact mod-0 binary
/// rows WIRED to their values by the reconstruction rows below.
const HP_DECOMP_BITS: usize = 639;
/// One exact (mod-0) reconstruction row per decomposed value.
const HP_DECOMP_RECON: usize = 5;

/// Number of group exponentiations per MultiSwap proof.
const N_GROUP_EXPS: usize = 4;
/// Group mults per MultiSwap proof.
const N_GROUP_MULS: usize = 2;

/// Exponent bit-length for the Fiat-Shamir prime challenge `ℓ`.
const ELL_BITS: usize = 352;

#[derive(Clone, Copy)]
struct Dims {
  /// Faithful-cost hash extension: chained Poseidon-cost rows,
  /// decomposition bit rows, and reconstruction rows for `Hp`.
  poseidon_rows: usize,
  decomp_bits: usize,
  decomp_recon: usize,
  k: usize,
  ell_bits: usize,
  n_group_exps: usize,
  n_group_muls: usize,
  hp_exps: usize,
  hp_exp_bits: usize,
  h_rows: usize,
}

impl Dims {
  fn multiswap(k: usize) -> Self {
    Self {
      k,
      poseidon_rows: HP_POSEIDON_PERMS * POSEIDON_ROWS_PER_PERM,
      decomp_bits: HP_DECOMP_BITS,
      decomp_recon: HP_DECOMP_RECON,
      ell_bits: ELL_BITS,
      n_group_exps: N_GROUP_EXPS,
      n_group_muls: N_GROUP_MULS,
      hp_exps: HP_EXPS,
      hp_exp_bits: HP_EXP_BITS,
      h_rows: H_ROWS,
    }
  }

  /// Rows of one wired Hp certificate chain.
  fn rows_per_hp_exp(&self) -> usize {
    3 * self.hp_exp_bits + 1
  }

  fn rows_per_exp(&self) -> usize {
    3 * self.ell_bits + 1
  }

  fn cols_per_exp(&self) -> usize {
    3 * self.ell_bits + 1
  }

  /// Unwired generic rows remaining: only the per-swap `H∆` models
  /// (k > 0). Everything at k = 0 is wired.
  fn generic_rows(&self) -> usize {
    2 * self.k + 2 * self.k * self.h_rows
  }

  /// Wired hash/Hp rows: group mults (operands = exp outputs, 1 fresh
  /// result column each), 4 Hp certificate chains, the Poseidon seed
  /// reduction row, the chained Poseidon rows, the decomposition bit
  /// rows, their reconstruction rows (0 fresh columns), and the final
  /// mod-ℓ reduction row wired to the Poseidon output.
  fn wired_ext_rows(&self) -> usize {
    self.n_group_muls
      + self.hp_exps * self.rows_per_hp_exp()
      + 1
      + self.poseidon_rows
      + self.decomp_bits
      + self.decomp_recon
      + 1
  }

  fn wired_ext_cols(&self) -> usize {
    self.n_group_muls
      + self.hp_exps * self.rows_per_hp_exp()
      + 1
      + self.poseidon_rows
      + self.decomp_bits
      + 1
  }

  fn non_exp_rows(&self) -> usize {
    self.generic_rows() + self.wired_ext_rows()
  }

  fn num_real_rows(&self) -> usize {
    self.n_group_exps * self.rows_per_exp() + self.non_exp_rows()
  }

  fn num_real_cols(&self) -> usize {
    self.n_group_exps * self.cols_per_exp() + 3 * self.generic_rows() + self.wired_ext_cols()
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

/// Moduli for the wired Hp certificate chains: Mersenne primes
/// 2^61−1, 2^89−1, 2^107−1, 2^127−1 (clean fixed constants standing in
/// for a Pocklington prime chain of growing widths).
fn hp_moduli() -> [BigUint; 4] {
  [
    (BigUint::from(1u32) << 61) - 1u32,
    (BigUint::from(1u32) << 89) - 1u32,
    (BigUint::from(1u32) << 107) - 1u32,
    (BigUint::from(1u32) << 127) - 1u32,
  ]
}

/// Synthetic (base, exponent) pairs for the Hp chains — deterministic,
/// bounded by each chain modulus / the chain bit width.
fn hp_chain_inputs(bits: usize) -> [(BigUint, BigUint); 4] {
  let ms = hp_moduli();
  core::array::from_fn(|i| {
    let base = &ms[i] - BigUint::from(1000u32 + 37 * i as u32);
    let exponent = (BigUint::from(0x9e37_79b9_7f4a_7c15u64) >> (64 - bits)) ^ BigUint::from(i);
    (base, exponent)
  })
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

  let mut exp_outs = Vec::with_capacity(d.n_group_exps);
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
    exp_outs.push(acc.clone());
    out.push(acc);
  }

  // Wired group mults from the exponentiation outputs.
  for i in 0..d.n_group_muls {
    out.push((&exp_outs[2 * i] * &exp_outs[2 * i + 1]).div_rem(n).1);
  }

  // Wired Hp certificate chains (square-and-multiply mod the Mersenne
  // moduli).
  let hp_ms = hp_moduli();
  let hp_inputs = hp_chain_inputs(d.hp_exp_bits);
  for i in 0..d.hp_exps {
    let g_minus_1 = &hp_inputs[i].0 - &one;
    let mut acc = one.clone();
    for j in 0..d.hp_exp_bits {
      let bit_pos = d.hp_exp_bits - 1 - j;
      let bit = u8::from(hp_inputs[i].1.bit(bit_pos as u64));
      let sq = (&acc * &acc).div_rem(&hp_ms[i]).1;
      let b_val = BigUint::from(bit) * &g_minus_1 + &one;
      acc = (&sq * &b_val).div_rem(&hp_ms[i]).1;
    }
    out.push(acc);
  }

  // Poseidon seeded from the first exponentiation output, then the
  // chained S-box values; final mod-ℓ reduction of the hash output.
  // (Bit rows need no divmods.)
  let mut x = exp_outs[0].div_rem(p_hash).1;
  for _ in 0..(d.poseidon_rows / 3) {
    let x2 = (&x * &x).div_rem(p_hash).1;
    let x4 = (&x2 * &x2).div_rem(p_hash).1;
    let x5 = (&x4 * &x).div_rem(p_hash).1;
    out.push(x2);
    out.push(x4);
    out.push(x5.clone());
    x = x5;
  }
  out.push(x.div_rem(ell).1);

  // Per-swap `H∆` models (k > 0 only).
  let groups: &[(&BigUint, usize)] = &[(ell, 2 * d.k), (p_hash, 2 * d.k * d.h_rows)];
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

/// Which statement the bench proves. `MSCFG=full` (default): the OWWB20
/// `SetBench` statement from `limber::multiswap` with real dataflow
/// (`MSSWAPS` swaps, default 1). `MSCFG=paper`: the cost-model circuit
/// above (the paper's submission-time row). `MSCFG=bare`: the four variable-base 352-bit
/// exponentiations of the Garuda / Zinc+ comparison rows. Returns
/// `(shape, w, q, public_io)`.
type Workload<MM> = (
  IntModR1CSShapeModp<MM>,
  Vec<BigUint>,
  Vec<BigUint>,
  Vec<BigUint>,
);
fn ws<MM: limber::traits::mod_engine::ModEngine>(d: Dims) -> Workload<MM> {
  use limber::multiswap::{
    poseidon::PoseidonParams,
    statement::{self, Config},
  };
  let cfg = cfg_name();
  match cfg.as_str() {
    "full" | "rsa" | "bare" => {
      let config = if cfg == "full" {
        let swaps = std::env::var("MSSWAPS")
          .ok()
          .and_then(|v| v.parse().ok())
          .unwrap_or(1);
        Config::Full { swaps }
      } else {
        Config::Rsa
      };
      let st = statement::build::<MM>(
        &config,
        &PoseidonParams::bls12_381_owwb20(),
        std::env::var_os("MSSEG").is_some(),
      )
      .expect("statement builds");
      (st.built.shape, st.built.w, st.built.q, st.built.io)
    }
    _ => {
      let (shape, w, q) = multiswap_shape_and_witness_for::<MM>(d);
      (shape, w, q, vec![])
    }
  }
}

fn cfg_name() -> String {
  std::env::var("MSCFG").unwrap_or_else(|_| "full".to_string())
}

fn multiswap_shape_and_witness_for<MM: limber::traits::mod_engine::ModEngine>(
  d: Dims,
) -> (IntModR1CSShapeModp<MM>, Vec<BigUint>, Vec<BigUint>) {
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

  // Wired group mults: operands are the exponentiation outputs
  // (Q^ℓ-style products), one fresh result column each.
  let exp_out = |i: usize| i * d.cols_per_exp() + 2 * d.ell_bits;
  for i in 0..d.n_group_muls {
    let a_col = exp_out(2 * i);
    let b_col = exp_out(2 * i + 1);
    let (qi, ci) = (&w[a_col] * &w[b_col]).div_rem(&n);
    w[col] = ci;
    q[row] = qi;
    a_entries.push((row, a_col, one.clone()));
    b_entries.push((row, b_col, one.clone()));
    c_entries.push((row, col, one.clone()));
    mods.push(n.clone());
    row += 1;
    col += 1;
  }

  // Wired Hp certificate chains: real square-and-multiply over the
  // Mersenne moduli, with bit decomposition and reconstruction —
  // structurally identical to the main Wesolowski chains.
  let hp_ms = hp_moduli();
  let hp_inputs = hp_chain_inputs(d.hp_exp_bits);
  let mut hp_out_cols = [0usize; 4];
  for i in 0..d.hp_exps {
    build_exp_circuit(
      &hp_inputs[i].0,
      &hp_inputs[i].1,
      &hp_ms[i],
      d.hp_exp_bits,
      row,
      col,
      const_col,
      &mut a_entries,
      &mut b_entries,
      &mut c_entries,
      &mut mods,
      &mut w,
      &mut q,
    );
    hp_out_cols[i] = col + 2 * d.hp_exp_bits;
    row += d.rows_per_hp_exp();
    col += d.rows_per_hp_exp();
  }

  // Poseidon seed: reduce the first exponentiation output mod p_hash —
  // the hash input is wired to real circuit data.
  let seed_col = col;
  {
    let (qi, ci) = w[exp_out(0)].div_rem(&p_hash);
    w[seed_col] = ci;
    q[row] = qi;
    a_entries.push((row, exp_out(0), one.clone()));
    b_entries.push((row, const_col, one.clone()));
    c_entries.push((row, seed_col, one.clone()));
    mods.push(p_hash.clone());
    row += 1;
    col += 1;
  }

  // Chained Poseidon-cost rows mod p_hash: per S-box x² = x·x,
  // x⁴ = x²·x², x⁵ = x⁴·x — one fresh column per row, the x⁵ output
  // feeding the next S-box (the linear MDS/round-constant layers fold
  // into the LCs of the following rows for free, exactly as a
  // constants-faithful build would).
  let zero = BigUint::from(0u32);
  let mut x_col = seed_col;
  for _ in 0..(d.poseidon_rows / 3) {
    let x = w[x_col].clone();
    let (q2, x2) = (&x * &x).div_rem(&p_hash);
    let (q4, x4) = (&x2 * &x2).div_rem(&p_hash);
    let (q5, x5) = (&x4 * &x).div_rem(&p_hash);
    // x² = x·x
    w[col] = x2;
    a_entries.push((row, x_col, one.clone()));
    b_entries.push((row, x_col, one.clone()));
    c_entries.push((row, col, one.clone()));
    mods.push(p_hash.clone());
    q[row] = q2;
    row += 1;
    // x⁴ = x²·x²
    w[col + 1] = x4;
    a_entries.push((row, col, one.clone()));
    b_entries.push((row, col, one.clone()));
    c_entries.push((row, col + 1, one.clone()));
    mods.push(p_hash.clone());
    q[row] = q4;
    row += 1;
    // x⁵ = x⁴·x
    w[col + 2] = x5;
    a_entries.push((row, col + 1, one.clone()));
    b_entries.push((row, x_col, one.clone()));
    c_entries.push((row, col + 2, one.clone()));
    mods.push(p_hash.clone());
    q[row] = q5;
    row += 1;
    x_col = col + 2;
    col += 3;
  }
  let pos_out_col = x_col;

  // Decomposition bit rows WIRED to real values: fully decompose the
  // Poseidon output and the four Hp chain outputs; each value gets an
  // exact (mod-0) reconstruction row referencing its bit columns —
  // zero fresh columns for reconstruction.
  let decomp_targets: Vec<(usize, usize)> = std::iter::once((pos_out_col, 255))
    .chain((0..4).map(|i| (hp_out_cols[i], [61usize, 89, 107, 127][i])))
    .collect();
  debug_assert_eq!(
    decomp_targets.iter().map(|&(_, b)| b).sum::<usize>(),
    d.decomp_bits
  );
  for &(val_col, nbits) in &decomp_targets {
    let val = w[val_col].clone();
    let bit_base = col;
    for j in 0..nbits {
      let bit = u8::from(val.bit((nbits - 1 - j) as u64));
      w[col] = BigUint::from(bit);
      a_entries.push((row, col, one.clone()));
      b_entries.push((row, col, one.clone()));
      c_entries.push((row, col, one.clone()));
      mods.push(zero.clone());
      q[row] = zero.clone();
      row += 1;
      col += 1;
    }
    for j in 0..nbits {
      let power = BigUint::from(1u32) << (nbits - 1 - j);
      a_entries.push((row, bit_base + j, power));
    }
    b_entries.push((row, const_col, one.clone()));
    c_entries.push((row, val_col, one.clone()));
    mods.push(zero.clone());
    q[row] = zero.clone();
    row += 1;
  }

  // Final mod-ℓ reduction row, wired to the Poseidon output.
  {
    let (qi, ci) = w[pos_out_col].div_rem(&ell);
    w[col] = ci;
    q[row] = qi;
    a_entries.push((row, pos_out_col, one.clone()));
    b_entries.push((row, const_col, one.clone()));
    c_entries.push((row, col, one.clone()));
    mods.push(ell.clone());
    row += 1;
    col += 1;
  }

  // Per-swap `H∆` models (k > 0 only): still generic 3-column rows,
  // flagged as unfaithful — do not quote k > 0 configurations.
  let groups: Vec<(&BigUint, usize)> = vec![(&ell, 2 * d.k), (&p_hash, 2 * d.k * d.h_rows)];
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
  debug_assert_eq!(row, d.num_real_rows());
  debug_assert_eq!(col, d.num_real_cols());

  mods.resize(num_cons, BigUint::from(2u32));

  let shape = IntModR1CSShapeModp::<MM>::new(
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

/// IntEval `k` for the Hyrax instantiation: `IMOD_K=<k>` overrides the
/// tuned `DEFAULT_K` (mirrors `BDK` for the Brakedown path).
fn hyrax_k() -> usize {
  std::env::var("IMOD_K")
    .ok()
    .and_then(|v| v.parse().ok())
    .unwrap_or(DEFAULT_K)
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

  // BDPCS=1: measure the Brakedown-backed instantiation (hash
  // commitments, non-hiding) on the same workload: commit+prove,
  // verify, and serialized proof size. The comparison point against
  // code-commitment systems.
  if std::env::var_os("BDPCS").is_some() {
    use limber::provider::T256DynPrimeBdEngine as BE;
    use std::time::Instant;
    if std::env::var_os("RUST_LOG").is_some() {
      let _ = tracing_subscriber::fmt()
        .with_target(false)
        .with_env_filter(tracing_subscriber::EnvFilter::from_default_env())
        .try_init();
    }
    let dims = Dims::multiswap(0);
    let (shape, w, q, io) = ws::<BE>(dims);
    let log_n = (shape.num_vars().max(shape.num_cons()) as u64).ilog2() as usize;
    let bdk: usize = std::env::var("BDK")
      .ok()
      .and_then(|v| v.parse().ok())
      .unwrap_or(11); // k=11 dominates k=9 across the sweep for the hash backend
    let params =
      IntEvalParams::derive(2048, LOG_T, bdk, log_n).expect("IntEval params satisfy bounds");
    let (pk, vk) = IntModSpartanModpSNARK::<BE>::setup_with_params(shape.clone(), params).unwrap();
    // Pre-warm the per-length code layouts (deterministic public
    // matrices; conceptually part of setup, not of commit).
    let tw = Instant::now();
    let nvars = shape.num_vars().max(shape.num_cons());
    let f_chunk_len = (nvars * 32 * 4).next_power_of_two();
    let _ = limber::provider::pcs::prewarm_brakedown_params(f_chunk_len);
    println!(
      "  (params prewarm for f-chunk length: {:.1} ms)",
      tw.elapsed().as_secs_f64() * 1e3
    );
    let t0 = Instant::now();
    let (witness, instance) = IntModR1CSWitnessModp::<BE>::new(&shape, pk.ck(), w, q, io).unwrap();
    let t_commit = t0.elapsed().as_secs_f64() * 1e3;
    let t1 = Instant::now();
    let proof = IntModSpartanModpSNARK::<BE>::prove(&pk, &instance, &witness).unwrap();
    let t_prove = t1.elapsed().as_secs_f64() * 1e3;
    let proof_bytes = proof.eval_arg_bytes().expect("eval_arg serializes").len();
    let t2 = Instant::now();
    proof.verify(&vk, &instance).unwrap();
    let t_verify = t2.elapsed().as_secs_f64() * 1e3;
    if let Some(path) = std::env::var_os("BDDUMP") {
      let bytes = bincode::serialize(proof.eval_arg_ref()).unwrap();
      std::fs::write(&path, &bytes).unwrap();
      println!("  dumped eval_arg ({} bytes) to {:?}", bytes.len(), path);
    }
    if std::env::var_os("BDANATOMY").is_some() {
      let open_args = proof.bd_open_args();
      for (g, a) in open_args.groups.iter().enumerate() {
        let (rows, cols, auth) = a.component_sizes();
        println!("  group {g}: combined rows {rows} B, columns {cols} B, auth {auth} B");
      }
      for (d, a) in open_args.direct.iter().enumerate() {
        println!("  direct {d}: {} B", a.size());
      }
    }
    println!(
      "MultiSwap {} 2^{} / Brakedown Mod-PCS: commit {t_commit:.1} ms, prove {t_prove:.1} ms, \
       total {:.1} ms, verify {t_verify:.1} ms, proof {} bytes ({:.2} MB)",
      cfg_name(),
      log_n,
      t_commit + t_prove,
      proof_bytes,
      proof_bytes as f64 / 1e6,
    );
    return;
  }

  // PSIZE=1: serialized proof size of the Hyrax-backed instantiation on
  // the standard workload. `eval_arg_bytes` covers the Mod-PCS batch
  // argument (commitments, GKR, combined opening) — the dominant part;
  // the dynamic-prime side (outer/inner sumcheck round polynomials and
  // claimed evals, ~1.2 KB at 2^13) is not yet `Serialize` and is
  // reported analytically alongside.
  // M127=1: the small-field instantiation (F127 + Brakedown) on the
  // standard workload — first honest numbers for the fast-prover
  // operating point. Unoptimized: eager delayed-reduction, per-target
  // Brakedown openings (no two-tree batching yet).
  if std::env::var_os("M127").is_some() {
    use std::time::Instant;
    if std::env::var_os("RUST_LOG").is_some() {
      let _ = tracing_subscriber::fmt()
        .with_target(false)
        .with_env_filter(tracing_subscriber::EnvFilter::from_default_env())
        .try_init();
    }
    type Sf = limber::provider::M127DynPrimeBdEngine;
    let dims = Dims::multiswap(0);
    let (shape, w, q, io) = ws::<Sf>(dims);
    let log_n = (shape.num_vars().max(shape.num_cons()) as u64).ilog2() as usize;
    // q = 127; 16-bit limbs (= chunks); k = 5 per the parameter grid.
    let params =
      IntEvalParams::derive_for_q(127, 2048, 16, 5, log_n).expect("M127 params satisfy bounds");
    println!(
      "M127 params: log_p={} s={} k={} numlimb={}",
      params.log_p, params.s, params.k, params.numlimb
    );
    let (pk, vk) = IntModSpartanModpSNARK::<Sf>::setup_with_params(shape.clone(), params).unwrap();
    let t0 = Instant::now();
    let (witness, instance) = IntModR1CSWitnessModp::<Sf>::new(&shape, pk.ck(), w, q, io).unwrap();
    let proof = IntModSpartanModpSNARK::<Sf>::prove(&pk, &instance, &witness).unwrap();
    let t_prove = t0.elapsed().as_secs_f64();
    let t1 = Instant::now();
    proof.verify(&vk, &instance).unwrap();
    let t_verify = t1.elapsed().as_secs_f64() * 1e3;
    let (pp, rc, co) = proof.eval_arg_component_sizes();
    println!(
      "MultiSwap 2^13 / M127-Brakedown: commit+prove {t_prove:.2} s, verify {t_verify:.1} ms, \
       eval_arg {:.2} MB [per_poly {pp} B, range_check {rc} B, combined_open {co} B]",
      (pp + rc + co) as f64 / 1e6,
    );
    return;
  }

  if std::env::var_os("PSIZE").is_some() {
    use std::time::Instant;
    if std::env::var_os("RUST_LOG").is_some() {
      let _ = tracing_subscriber::fmt()
        .with_target(false)
        .with_env_filter(tracing_subscriber::EnvFilter::from_default_env())
        .try_init();
    }
    let dims = Dims::multiswap(0);
    let (shape, w, q, io) = ws::<M>(dims);
    let log_n = (shape.num_vars().max(shape.num_cons()) as u64).ilog2() as usize;
    let params =
      IntEvalParams::derive(2048, LOG_T, hyrax_k(), log_n).expect("IntEval params satisfy bounds");
    let (pk, vk) = IntModSpartanModpSNARK::<M>::setup_with_params(shape.clone(), params).unwrap();
    let t0 = Instant::now();
    let (witness, instance) = IntModR1CSWitnessModp::<M>::new(&shape, pk.ck(), w, q, io).unwrap();
    let proof = IntModSpartanModpSNARK::<M>::prove(&pk, &instance, &witness).unwrap();
    let total = t0.elapsed().as_secs_f64();
    let t1 = Instant::now();
    proof.verify(&vk, &instance).unwrap();
    let verify_ms = t1.elapsed().as_secs_f64() * 1e3;
    let arg_bytes = proof.eval_arg_size();
    let (pp, rc, co) = proof.eval_arg_component_sizes();
    println!("  breakdown: per_poly {pp} B, range_check {rc} B, combined_open {co} B");
    // PSDUMP=<path>: write the serialized eval_arg so its compressed
    // size can be measured externally (e.g. `zstd -19`).
    if let Some(path) = std::env::var_os("PSDUMP") {
      let bytes = bincode::serialize(proof.eval_arg_ref()).unwrap();
      std::fs::write(&path, &bytes).unwrap();
      println!("  dumped eval_arg ({} bytes) to {:?}", bytes.len(), path);
    }
    // Dynamic-prime remainder: 13 cubic outer rounds (3 coeffs each) +
    // 14 quadratic inner rounds (2 coeffs) + 6 claimed evals, 16 B per
    // 2-limb scalar.
    let dyn_bytes = (13 * 3 + 14 * 2 + 6) * 16;
    println!(
      "MultiSwap {} 2^{} / Hyrax Mod-PCS proof size: eval_arg {arg_bytes} bytes \
       + ~{dyn_bytes} B sumcheck side ≈ {:.1} KB  (commit+prove {total:.2} s, verify {verify_ms:.1} ms)",
      cfg_name(),
      log_n,
      (arg_bytes + dyn_bytes) as f64 / 1e3,
    );
    return;
  }

  // KSWEEP=1: time (commit + prove) per k at the current LOG_T, verify each
  // (soundness gate), print, and return (skip criterion). Used to re-find the
  // optimal k after a LOG_T change.
  if std::env::var_os("KSWEEP").is_some() {
    use std::time::Instant;
    let dims = Dims::multiswap(0);
    let (shape, w, q, io) = ws::<M>(dims);
    println!(
      "\nMultiSwap k-sweep (LOG_T={LOG_T}, 2^{} rows):",
      (shape.num_cons() as u64).ilog2(),
    );
    for k in [9usize, 10, 11, 12, 13] {
      let params = params_for(&shape, k);
      let (sval, lpval, nl) = (params.s, params.log_p, params.numlimb);
      let (pk, vk) = IntModSpartanModpSNARK::<M>::setup_with_params(shape.clone(), params).unwrap();
      let t0 = Instant::now();
      let (witness, instance) =
        IntModR1CSWitnessModp::<M>::new(&shape, pk.ck(), w.clone(), q.clone(), io.clone()).unwrap();
      let proof = IntModSpartanModpSNARK::<M>::prove(&pk, &instance, &witness).unwrap();
      let ms = t0.elapsed().as_secs_f64() * 1e3;
      proof.verify(&vk, &instance).unwrap();
      println!("  k={k:<2} (s={sval}, log_p={lpval}, numlimb={nl}): commit+prove {ms:8.1} ms");
    }
    return;
  }

  if std::env::var_os("RUST_LOG").is_some() {
    let _ = tracing_subscriber::fmt()
      .with_target(false)
      .with_env_filter(EnvFilter::from_default_env())
      .try_init();
    for &k in ks {
      let dims = Dims::multiswap(k);
      let (shape, w, q, io) = ws::<M>(dims);
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
          IntModR1CSWitnessModp::<M>::new(&shape, pk.ck(), w.clone(), q.clone(), io.clone())
            .unwrap();
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
    let (shape, _w, _q, _io) = ws::<M>(dims);
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
    let (shape0, _, _, _) = ws::<M>(dims);
    let tag = format!(
      "{}_k{k}_c2^{}",
      cfg_name(),
      (shape0.num_cons() as u64).ilog2()
    );

    g.bench_function(format!("setup/{tag}"), |b| {
      b.iter_batched(
        || {
          let (shape, _, _, _) = ws::<M>(dims);
          let params = params_for(&shape, hyrax_k());
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
          let (shape, w, q, io) = ws::<M>(dims);
          let params = params_for(&shape, hyrax_k());
          let (pk, _vk) =
            IntModSpartanModpSNARK::<M>::setup_with_params(shape.clone(), params).unwrap();
          (pk, shape, w, q, io)
        },
        |(pk, shape, w, q, io)| {
          let _ = IntModR1CSWitnessModp::<M>::new(&shape, pk.ck(), w, q, io).unwrap();
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
          let (shape, _, _, _) = ws::<M>(dims);
          let params = params_for(&shape, hyrax_k());
          let (pk, _vk) = IntModSpartanModpSNARK::<M>::setup_with_params(shape, params).unwrap();
          pk
        },
        |pk| {
          let (shape, w, q, io) = ws::<M>(dims);
          let (witness, instance) =
            IntModR1CSWitnessModp::<M>::new(&shape, pk.ck(), w, q, io).unwrap();
          let _ = IntModSpartanModpSNARK::<M>::prove(&pk, &instance, &witness).unwrap();
        },
        BatchSize::LargeInput,
      );
    });

    g.bench_function(format!("verify/{tag}"), |b| {
      b.iter_batched(
        || {
          let (shape, w, q, io) = ws::<M>(dims);
          let params = params_for(&shape, hyrax_k());
          let (pk, vk) =
            IntModSpartanModpSNARK::<M>::setup_with_params(shape.clone(), params).unwrap();
          let (witness, instance) =
            IntModR1CSWitnessModp::<M>::new(&shape, pk.ck(), w, q, io).unwrap();
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
