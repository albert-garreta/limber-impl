//! A row builder for Integer Mod-R1CS circuits over symbolic linear
//! combinations, with the gadgets the MultiSwap statement needs.
//!
//! Every row is `⟨A, z⟩ · ⟨B, z⟩ = ⟨C, z⟩ + m · q` over the non-negative
//! integers with a public modulus `m` (`m = 0` means exact). All matrix
//! coefficients are non-negative, so a subtraction is always expressed
//! as a witnessed difference (`d + b = a`, exact), and "`x ≥ 0`" for a
//! witness is free — the Mod-PCS bounds every committed value. Values
//! asserted below `2^16` live in a dedicated [`SmallValueBlock`] at the
//! start of the witness and cost no rows.
//!
//! The builder evaluates every row as it is emitted and panics on an
//! unsatisfied one, so a circuit that finalizes is satisfied by
//! construction; `IntModR1CSShapeModp::is_sat` re-checks independently.

use crate::{
  errors::SpartanError,
  imod_r1cs_modp::IntModR1CSShapeModp,
  traits::mod_engine::{ModEngine, SmallValueBlock},
};
use num_bigint::BigUint;
use num_integer::Integer;
use num_traits::{One, Zero};

/// A symbolic variable of `z = (w, 1, x)`.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Var {
  /// Witness column (regular region).
  W(usize),
  /// Witness column in the small-value block.
  Small(usize),
  /// The constant-one column.
  Const,
  /// Public input `x[i]`.
  Io(usize),
}

/// A linear combination `Σ coeff · var` with non-negative coefficients.
#[derive(Clone, Debug, Default)]
pub struct Lc(pub Vec<(Var, BigUint)>);

impl Lc {
  /// The constant `c`.
  pub fn constant(c: BigUint) -> Self {
    if c.is_zero() {
      Self::default()
    } else {
      Self(vec![(Var::Const, c)])
    }
  }
  /// A single variable.
  pub fn var(v: Var) -> Self {
    Self(vec![(v, BigUint::one())])
  }
  /// `self + coeff · v`.
  pub fn add_term(mut self, v: Var, coeff: BigUint) -> Self {
    if !coeff.is_zero() {
      self.0.push((v, coeff));
    }
    self
  }
  /// `self + other`.
  pub fn plus(mut self, other: &Lc) -> Self {
    self.0.extend(other.0.iter().cloned());
    self
  }
  /// `self + c`.
  pub fn add_const(self, c: &BigUint) -> Self {
    self.add_term(Var::Const, c.clone())
  }
  /// `k · self`.
  pub fn scale(&self, k: &BigUint) -> Self {
    Self(self.0.iter().map(|(v, c)| (*v, c * k)).collect())
  }
  /// Merge duplicate variables and reduce coefficients modulo `p` (only
  /// valid for combinations consumed by rows with modulus `p`).
  pub fn normalize_mod(&self, p: &BigUint) -> Self {
    let mut acc: Vec<(Var, BigUint)> = Vec::new();
    for (v, c) in &self.0 {
      if let Some(slot) = acc.iter_mut().find(|(u, _)| u == v) {
        slot.1 = (&slot.1 + c) % p;
      } else {
        acc.push((*v, c % p));
      }
    }
    acc.retain(|(_, c)| !c.is_zero());
    Self(acc)
  }
  /// Number of terms.
  pub fn len(&self) -> usize {
    self.0.len()
  }
  /// Whether the combination is empty (identically zero).
  pub fn is_empty(&self) -> bool {
    self.0.is_empty()
  }
}

/// Which modulus a row reduces by.
#[derive(Clone, Debug)]
pub enum Modulus {
  /// Exact integer equation.
  Exact,
  /// Public constant modulus.
  Public(BigUint),
}

/// The builder state: rows, column values, and the small-value block.
#[derive(Debug, Default)]
pub struct Builder {
  /// Regular witness values.
  pub w: Vec<BigUint>,
  /// Small-value block values (each `< 2^16`).
  pub small: Vec<BigUint>,
  /// Public inputs.
  pub io: Vec<BigUint>,
  a: Vec<(usize, Lc)>,
  b: Vec<(usize, Lc)>,
  c: Vec<(usize, Lc)>,
  mods: Vec<BigUint>,
  q: Vec<BigUint>,
  /// Non-zero matrix entries emitted so far (after LC merging).
  pub nnz: usize,
}

/// A finalized circuit: shape plus its satisfying assignment.
pub struct Built<M: ModEngine> {
  /// The shape (with the small-value block declared).
  pub shape: IntModR1CSShapeModp<M>,
  /// Witness `w` (padded).
  pub w: Vec<BigUint>,
  /// Quotients `q` (padded).
  pub q: Vec<BigUint>,
  /// Public inputs.
  pub io: Vec<BigUint>,
  /// Real (unpadded) row count.
  pub real_rows: usize,
  /// Real (unpadded) column count, block included.
  pub real_cols: usize,
  /// Size of the small-value block (a power of two, possibly zero).
  pub block_len: usize,
}

impl Builder {
  /// Empty builder.
  pub fn new() -> Self {
    Self::default()
  }

  /// Rows emitted so far.
  pub fn num_rows(&self) -> usize {
    self.mods.len()
  }

  /// Value of a variable.
  pub fn value(&self, v: Var) -> BigUint {
    match v {
      Var::W(i) => self.w[i].clone(),
      Var::Small(i) => self.small[i].clone(),
      Var::Const => BigUint::one(),
      Var::Io(i) => self.io[i].clone(),
    }
  }

  /// Value of a linear combination.
  pub fn eval(&self, lc: &Lc) -> BigUint {
    let mut acc = BigUint::zero();
    for (v, c) in &lc.0 {
      acc += c * self.value(*v);
    }
    acc
  }

  /// Allocate a regular witness.
  pub fn alloc(&mut self, v: BigUint) -> Var {
    self.w.push(v);
    Var::W(self.w.len() - 1)
  }

  /// Allocate a small-value witness (`v < 2^16`, enforced by the block).
  pub fn alloc_small(&mut self, v: BigUint) -> Var {
    assert!(v.bits() <= 16, "alloc_small: value has {} bits", v.bits());
    self.small.push(v);
    Var::Small(self.small.len() - 1)
  }

  /// Allocate a public input.
  pub fn alloc_io(&mut self, v: BigUint) -> Var {
    self.io.push(v);
    Var::Io(self.io.len() - 1)
  }

  /// Emit a row `a · b = c (mod m)`, computing and checking the quotient.
  pub fn row(&mut self, a: Lc, b: Lc, c: Lc, m: Modulus) {
    let av = self.eval(&a);
    let bv = self.eval(&b);
    let cv = self.eval(&c);
    let lhs = &av * &bv;
    let (m_val, q) = match &m {
      Modulus::Exact => {
        assert!(
          lhs == cv,
          "row {}: exact row unsatisfied ({lhs} != {cv})",
          self.num_rows()
        );
        (BigUint::zero(), BigUint::zero())
      }
      Modulus::Public(m) => {
        assert!(
          lhs >= cv,
          "row {}: lhs {lhs} < rhs {cv} (quotient would be negative)",
          self.num_rows()
        );
        let diff = &lhs - &cv;
        let (q, r) = diff.div_rem(m);
        assert!(
          r.is_zero(),
          "row {}: lhs − rhs not divisible by the modulus",
          self.num_rows()
        );
        (m.clone(), q)
      }
    };
    let i = self.num_rows();
    self.nnz += a.len() + b.len() + c.len();
    self.a.push((i, a));
    self.b.push((i, b));
    self.c.push((i, c));
    self.mods.push(m_val);
    self.q.push(q);
  }

  // ------------------------------------------------------------ gadgets

  /// `c = a · b mod m` (public modulus); returns the new witness.
  pub fn mul_mod(&mut self, a: &Lc, b: &Lc, m: &BigUint) -> Var {
    let v = (self.eval(a) * self.eval(b)) % m;
    let c = self.alloc(v);
    self.row(a.clone(), b.clone(), Lc::var(c), Modulus::Public(m.clone()));
    c
  }

  /// `c = a · b` exactly; returns the new witness.
  pub fn mul_exact(&mut self, a: &Lc, b: &Lc) -> Var {
    let v = self.eval(a) * self.eval(b);
    let c = self.alloc(v);
    self.row(a.clone(), b.clone(), Lc::var(c), Modulus::Exact);
    c
  }

  /// Assert `a · b = c` exactly.
  pub fn assert_exact(&mut self, a: &Lc, b: &Lc, c: &Lc) {
    self.row(a.clone(), b.clone(), c.clone(), Modulus::Exact);
  }

  /// Assert `lc = v mod m` for a fresh witness `v` (materialize an LC
  /// reduced modulo a public modulus).
  pub fn reduce_mod(&mut self, lc: &Lc, m: &BigUint) -> Var {
    let v = self.eval(lc) % m;
    let out = self.alloc(v);
    self.row(
      lc.clone(),
      Lc::constant(BigUint::one()),
      Lc::var(out),
      Modulus::Public(m.clone()),
    );
    out
  }

  /// Materialize an exact LC as a witness (`v = lc`).
  pub fn materialize(&mut self, lc: &Lc) -> Var {
    let v = self.eval(lc);
    let out = self.alloc(v);
    self.assert_exact(lc, &Lc::constant(BigUint::one()), &Lc::var(out));
    out
  }

  /// `d = a − b` (requires `a ≥ b`): witness `d` with `d + b = a`.
  pub fn sub(&mut self, a: &Lc, b: &Lc) -> Var {
    let av = self.eval(a);
    let bv = self.eval(b);
    assert!(av >= bv, "sub: negative difference");
    let d = self.alloc(&av - &bv);
    self.assert_exact(&Lc::var(d).plus(b), &Lc::constant(BigUint::one()), a);
    d
  }

  /// Assert `a < b`: witness `e` with `e + a + 1 = b`.
  pub fn assert_lt(&mut self, a: &Lc, b: &Lc) {
    let av = self.eval(a);
    let bv = self.eval(b);
    assert!(av < bv, "assert_lt: {av} >= {bv}");
    let e = self.alloc(&bv - &av - BigUint::one());
    self.assert_exact(
      &Lc::var(e).plus(a).add_const(&BigUint::one()),
      &Lc::constant(BigUint::one()),
      b,
    );
  }

  /// `c = a · b mod n` for a WITNESS modulus `n`: `t = n · k`, then
  /// `a · b = c + t` exactly. Two rows.
  pub fn mul_mod_witness(&mut self, a: &Lc, b: &Lc, n: &Lc) -> Var {
    let nv = self.eval(n);
    let prod = self.eval(a) * self.eval(b);
    let (k, c) = prod.div_rem(&nv);
    let k = self.alloc(k);
    let t = self.mul_exact(n, &Lc::var(k));
    let c = self.alloc(c);
    self.row(
      a.clone(),
      b.clone(),
      Lc::var(c).add_term(t, BigUint::one()),
      Modulus::Exact,
    );
    c
  }

  /// Allocate a bit with its exact `b · b = b` row.
  pub fn alloc_bit(&mut self, bit: bool) -> Var {
    let b = self.alloc(BigUint::from(bit as u8));
    self.assert_exact(&Lc::var(b), &Lc::var(b), &Lc::var(b));
    b
  }

  /// Decompose `lc` into `nbits` bits (LSB first) with bit rows and one
  /// exact reconstruction row. Panics if the value does not fit.
  pub fn bits_of(&mut self, lc: &Lc, nbits: usize) -> Vec<Var> {
    let v = self.eval(lc);
    assert!(v.bits() as usize <= nbits, "bits_of: value does not fit");
    let bits: Vec<Var> = (0..nbits as u64)
      .map(|i| self.alloc_bit(v.bit(i)))
      .collect();
    let mut recon = Lc::default();
    for (i, b) in bits.iter().enumerate() {
      recon = recon.add_term(*b, BigUint::one() << i);
    }
    self.assert_exact(&recon, &Lc::constant(BigUint::one()), lc);
    bits
  }

  /// Decompose `lc` into `nchunks` 16-bit chunks (LSB first) in the
  /// small-value block, with one exact reconstruction row.
  pub fn chunks_of(&mut self, lc: &Lc, nchunks: usize) -> Vec<Var> {
    let v = self.eval(lc);
    assert!(
      v.bits() as usize <= 16 * nchunks,
      "chunks_of: value does not fit"
    );
    let mask = BigUint::from(0xffffu32);
    let chunks: Vec<Var> = (0..nchunks)
      .map(|i| self.alloc_small((&v >> (16 * i)) & &mask))
      .collect();
    let mut recon = Lc::default();
    for (i, c) in chunks.iter().enumerate() {
      recon = recon.add_term(*c, BigUint::one() << (16 * i));
    }
    self.assert_exact(&recon, &Lc::constant(BigUint::one()), lc);
    chunks
  }

  /// `Σ 2^i · bits[i]` as an LC (LSB first).
  pub fn lc_of_bits(bits: &[Var]) -> Lc {
    let mut lc = Lc::default();
    for (i, b) in bits.iter().enumerate() {
      lc = lc.add_term(*b, BigUint::one() << i);
    }
    lc
  }

  /// Square-and-multiply `base^e mod m` with a VARIABLE base, `e` given
  /// as bits MSB first (each bit an LC — a witness bit or the constant
  /// 1). Per bit: `sq = acc² mod m`, `t = bit · (base − 1)`,
  /// `acc' = sq · (t + 1) mod m`. `base_minus_one` is the witnessed
  /// `base − 1` (from [`Self::sub`]). For a witness modulus pass
  /// `mod_lc = Some(n)`; each modular product then costs two rows.
  pub fn exp_var_base(
    &mut self,
    base: &Lc,
    base_minus_one: &Lc,
    exp_bits_msb: &[Lc],
    m: &BigUint,
    mod_lc: Option<&Lc>,
  ) -> Var {
    let mut acc = Lc::constant(BigUint::one());
    let mut acc_var: Option<Var> = None;
    for bit in exp_bits_msb {
      let sq = match mod_lc {
        Some(n) => self.mul_mod_witness(&acc, &acc, n),
        None => self.mul_mod(&acc, &acc, m),
      };
      let t = self.mul_exact(bit, base_minus_one);
      let mult = Lc::var(t).add_const(&BigUint::one());
      let next = match mod_lc {
        Some(n) => self.mul_mod_witness(&Lc::var(sq), &mult, n),
        None => self.mul_mod(&Lc::var(sq), &mult, m),
      };
      acc = Lc::var(next);
      acc_var = Some(next);
    }
    let _ = base;
    acc_var.expect("at least one exponent bit")
  }

  /// Square-and-multiply with a CONSTANT base `g` (the multiplier
  /// `bit · (g − 1) + 1` is a linear combination): 2 rows per bit.
  pub fn exp_const_base(&mut self, g: &BigUint, exp_bits_msb: &[Lc], m: &BigUint) -> Var {
    let g_minus_1 = g - BigUint::one();
    let mut acc = Lc::constant(BigUint::one());
    let mut acc_var: Option<Var> = None;
    for bit in exp_bits_msb {
      let sq = self.mul_mod(&acc, &acc, m);
      let mult = bit.scale(&g_minus_1).add_const(&BigUint::one());
      let next = self.mul_mod(&Lc::var(sq), &mult, m);
      acc = Lc::var(next);
      acc_var = Some(next);
    }
    acc_var.expect("at least one exponent bit")
  }

  /// Constant-base square-and-multiply modulo a WITNESS modulus `n`
  /// (four rows per bit: two witnessed-modulus products).
  pub fn exp_const_base_witness_mod(&mut self, g: &BigUint, exp_bits_msb: &[Lc], n: &Lc) -> Var {
    let g_minus_1 = g - BigUint::one();
    let mut acc = Lc::constant(BigUint::one());
    let mut acc_var: Option<Var> = None;
    for bit in exp_bits_msb {
      let sq = self.mul_mod_witness(&acc, &acc, n);
      let mult = bit.scale(&g_minus_1).add_const(&BigUint::one());
      let next = self.mul_mod_witness(&Lc::var(sq), &mult, n);
      acc = Lc::var(next);
      acc_var = Some(next);
    }
    acc_var.expect("at least one exponent bit")
  }

  /// Canonical representative `min(x, N − x)` of the quotient group
  /// `(Z/N)^*/{±1}`: `y = N − x`; selector bit `s = [y < x]`;
  /// `r = s·y + (1−s)·x` via two products and one exact row; `x − r ≥ 0`
  /// and `y − r ≥ 0` as witnessed differences. Six rows.
  pub fn canon(&mut self, x: &Lc, n: &BigUint) -> Var {
    let xv = self.eval(x);
    let y = self.sub(&Lc::constant(n.clone()), x);
    let yv = n - &xv;
    let s = self.alloc_bit(yv < xv);
    let t1 = self.mul_exact(&Lc::var(s), &Lc::var(y));
    let t2 = self.mul_exact(&Lc::var(s), x);
    let rv = if yv < xv { yv.clone() } else { xv.clone() };
    let r = self.alloc(rv);
    self.assert_exact(
      &Lc::var(t1).plus(x),
      &Lc::constant(BigUint::one()),
      &Lc::var(r).add_term(t2, BigUint::one()),
    );
    let _dx = self.sub(x, &Lc::var(r));
    let _dy = self.sub(&Lc::var(y), &Lc::var(r));
    r
  }

  /// Finalize into a shape and padded witness. `block_len_hint` lets the
  /// caller reserve a larger block than needed (kept a power of two).
  pub fn finalize<M: ModEngine>(self) -> Result<Built<M>, SpartanError> {
    let block_len = if self.small.is_empty() {
      0
    } else {
      self.small.len().next_power_of_two()
    };
    let real_cols = block_len + self.w.len();
    let num_vars = real_cols.next_power_of_two().max(2);
    let num_cons = self.num_rows().next_power_of_two().max(2);
    let num_io = self.io.len();
    let const_col = num_vars;
    let resolve = |v: Var| -> usize {
      match v {
        Var::Small(i) => i,
        Var::W(i) => block_len + i,
        Var::Const => const_col,
        Var::Io(i) => const_col + 1 + i,
      }
    };
    let to_entries = |rows: &[(usize, Lc)]| -> Vec<(usize, usize, BigUint)> {
      let mut out = Vec::new();
      for (r, lc) in rows {
        // Merge duplicate variables within an LC (exact sums).
        let mut acc: Vec<(usize, BigUint)> = Vec::new();
        for (v, c) in &lc.0 {
          let col = resolve(*v);
          if let Some(slot) = acc.iter_mut().find(|(u, _)| *u == col) {
            slot.1 += c;
          } else {
            acc.push((col, c.clone()));
          }
        }
        for (col, c) in acc {
          if !c.is_zero() {
            out.push((*r, col, c));
          }
        }
      }
      out
    };
    let a = to_entries(&self.a);
    let b = to_entries(&self.b);
    let c = to_entries(&self.c);
    let mut mods = self.mods;
    mods.resize(num_cons, BigUint::from(2u32));
    let mut w = vec![BigUint::zero(); num_vars];
    w[..self.small.len()].clone_from_slice(&self.small);
    w[block_len..block_len + self.w.len()].clone_from_slice(&self.w);
    let mut q = self.q;
    q.resize(num_cons, BigUint::zero());
    let mut shape = IntModR1CSShapeModp::<M>::new(num_cons, num_vars, num_io, a, b, c, mods)?;
    if block_len > 0 {
      shape = shape.with_small_value_blocks(vec![SmallValueBlock {
        start: 0,
        log_len: block_len.trailing_zeros() as usize,
      }])?;
    }
    Ok(Built {
      shape,
      w,
      q,
      io: self.io,
      real_rows: self.a.len(),
      real_cols,
      block_len,
    })
  }
}

#[cfg(test)]
mod tests {
  use super::*;
  use crate::provider::T256DynPrimeEngine;

  #[test]
  fn exp_gadgets_match_modpow() {
    let m = BigUint::from(1_000_003u64);
    let mut b = Builder::new();
    let g = BigUint::from(12345u64);
    let e = BigUint::from(0xdeadbeefu64);
    let e_lc = Lc::var(b.alloc(e.clone()));
    let bits = b.bits_of(&e_lc, 32);
    let bits_msb: Vec<Lc> = bits.iter().rev().map(|v| Lc::var(*v)).collect();
    let r1 = b.exp_const_base(&g, &bits_msb, &m);
    let gv = b.alloc(g.clone());
    let gm1 = b.sub(&Lc::var(gv), &Lc::constant(BigUint::one()));
    let r2 = b.exp_var_base(&Lc::var(gv), &Lc::var(gm1), &bits_msb, &m, None);
    let n = b.alloc(m.clone());
    let r3 = b.exp_var_base(
      &Lc::var(gv),
      &Lc::var(gm1),
      &bits_msb,
      &m,
      Some(&Lc::var(n)),
    );
    let expect = g.modpow(&e, &m);
    assert_eq!(b.value(r1), expect);
    assert_eq!(b.value(r2), expect);
    assert_eq!(b.value(r3), expect);
    let c = b.canon(&Lc::var(r2), &m);
    let cv = b.value(c);
    assert!(cv == expect || cv == &m - &expect);
    let built = b.finalize::<T256DynPrimeEngine>().unwrap();
    assert_eq!(built.io.len(), 0);
    assert!(built.real_rows > 32 * 5);
  }

  #[test]
  fn chunks_and_small_block_finalize() {
    let mut b = Builder::new();
    let v = b.alloc(BigUint::from(0x1234_5678_9abcu64));
    let ch = b.chunks_of(&Lc::var(v), 3);
    assert_eq!(b.value(ch[0]), BigUint::from(0x9abcu32));
    assert_eq!(b.value(ch[2]), BigUint::from(0x1234u32));
    let io = b.alloc_io(BigUint::from(7u32));
    let _ = b.materialize(&Lc::var(io).add_const(&BigUint::from(3u32)));
    let built = b.finalize::<T256DynPrimeEngine>().unwrap();
    assert_eq!(built.block_len, 4);
    assert_eq!(built.shape.small_value_blocks().len(), 1);
    assert_eq!(built.io, vec![BigUint::from(7u32)]);
  }
}
