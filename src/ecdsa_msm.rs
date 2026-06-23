//! ECDSA-verification MSM circuit gadgets for integer Mod-R1CS (secp256k1).
//!
//! Builds the elliptic-curve arithmetic of `R = u1·G + u2·Q` as rows of the
//! relation `A·z ∘ B·z = C·z + m∘q` (per-row modulus `m = p`, the secp256k1
//! base prime). EC coordinates are affine; divisions use prover-advice (one
//! row). Subtractions use **difference witnesses** to stay non-negative (a
//! `(p−1)`-coefficient negation would produce negative quotients — unsound here;
//! see `docs/ecdsa_benchmark_plan.md`). All witness/quotient values are `< p`.
//!
//! This is the gadget layer; the full Shamir MSM loop and bench build on it.

#![allow(dead_code)]

use num_bigint::BigUint;
use num_integer::Integer;

/// secp256k1 base field prime `p = 2^256 − 2^32 − 977`.
fn secp256k1_p() -> BigUint {
  BigUint::parse_bytes(
    b"FFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFEFFFFFC2F",
    16,
  )
  .unwrap()
}

/// Modular inverse via Fermat (`p` prime): `a^{p−2} mod p`.
fn mod_inv(a: &BigUint, p: &BigUint) -> BigUint {
  a.modpow(&(p - 2u32), p)
}

/// Reference affine EC addition over secp256k1 for distinct, non-identity
/// points `P1 + P2` (no doubling, no identity). Returns `(x3, y3)`.
fn ref_ec_add(
  x1: &BigUint,
  y1: &BigUint,
  x2: &BigUint,
  y2: &BigUint,
  p: &BigUint,
) -> (BigUint, BigUint) {
  let dx = (x2 + p - x1) % p;
  let dy = (y2 + p - y1) % p;
  let lam = (&dy * mod_inv(&dx, p)) % p;
  let t = (&lam * &lam) % p;
  let x3 = (&t + p + p - x1 - x2) % p;
  let dx3 = (x1 + p - &x3) % p;
  let u = (&lam * &dx3) % p;
  let y3 = (&u + p - y1) % p;
  (x3, y3)
}

/// A single (row, col, coeff) triple for one of the A/B/C matrices.
type Triple = (usize, usize, BigUint);

/// Accumulates the rows of an integer Mod-R1CS circuit as it is wired.
struct CircuitBuilder {
  a: Vec<Triple>,
  b: Vec<Triple>,
  c: Vec<Triple>,
  mods: Vec<BigUint>,
  w: Vec<BigUint>,
  q: Vec<BigUint>,
  /// next free witness column
  next_col: usize,
  /// the constant-1 column index (set when finalized); during building we
  /// record references and patch them — but here we pass it in explicitly.
  const_col: usize,
  p: BigUint,
}

impl CircuitBuilder {
  fn new(const_col: usize, p: BigUint) -> Self {
    let mut w = vec![BigUint::ZERO; const_col + 1];
    w[const_col] = BigUint::from(1u32); // not stored in w[] in the real shape, but
    // we keep a local z-style vector for the relation self-check in tests.
    Self {
      a: Vec::new(),
      b: Vec::new(),
      c: Vec::new(),
      mods: Vec::new(),
      w,
      q: Vec::new(),
      next_col: 0,
      const_col,
      p,
    }
  }

  /// Allocate a fresh witness column holding `val`.
  fn alloc(&mut self, val: BigUint) -> usize {
    let col = self.next_col;
    self.next_col += 1;
    if col >= self.w.len() {
      self.w.resize(col + 1, BigUint::ZERO);
    }
    self.w[col] = val;
    col
  }

  /// Push a modular-mult row `(Σa)·(Σb) = (Σc) + p·q` with modulus `p`,
  /// computing and storing the quotient. Panics (in tests) if `q` would be
  /// negative — the soundness guard.
  fn push_row(&mut self, row: usize, a: &[(usize, u32)], b: &[(usize, u32)], c: &[(usize, u32)]) {
    let lc = |terms: &[(usize, u32)], this: &Self| -> BigUint {
      terms
        .iter()
        .map(|(col, k)| BigUint::from(*k) * &this.w[*col])
        .sum()
    };
    let az = lc(a, self);
    let bz = lc(b, self);
    let cz = lc(c, self);
    let lhs = &az * &bz;
    assert!(lhs >= cz, "negative quotient at row {row} (unsound wiring)");
    let (qv, rem) = (&lhs - &cz).div_rem(&self.p);
    assert!(rem == BigUint::ZERO, "row {row} not satisfied mod p");
    for (col, k) in a {
      self.a.push((row, *col, BigUint::from(*k)));
    }
    for (col, k) in b {
      self.b.push((row, *col, BigUint::from(*k)));
    }
    for (col, k) in c {
      self.c.push((row, *col, BigUint::from(*k)));
    }
    self.mods.push(self.p.clone());
    if row >= self.q.len() {
      self.q.resize(row + 1, BigUint::ZERO);
    }
    self.q[row] = qv;
  }

  /// Difference witness `d = (a − b) mod p` with the row `d + b ≡ a (mod p)`.
  /// Returns the column of `d`.
  fn diff(&mut self, row: usize, a_col: usize, b_col: usize) -> usize {
    let a_val = self.w[a_col].clone();
    let b_val = self.w[b_col].clone();
    let d = (&a_val + &self.p - &b_val) % &self.p;
    let d_col = self.alloc(d);
    // (d + b)·1 = a + p·q
    self.push_row(
      row,
      &[(d_col, 1), (b_col, 1)],
      &[(self.const_col, 1)],
      &[(a_col, 1)],
    );
    d_col
  }

  /// Modular product witness `m = (x·y) mod p` with row `x·y = m + p·q`.
  fn mul(&mut self, row: usize, x_col: usize, y_col: usize) -> usize {
    let m = (&self.w[x_col] * &self.w[y_col]) % &self.p;
    let m_col = self.alloc(m);
    self.push_row(row, &[(x_col, 1)], &[(y_col, 1)], &[(m_col, 1)]);
    m_col
  }

  /// Affine EC add of points at columns `(x1,y1)`,`(x2,y2)`. Returns `(x3,y3)`
  /// columns. Uses 9 rows starting at `row`; returns the next free row.
  fn ec_add(
    &mut self,
    mut row: usize,
    x1: usize,
    y1: usize,
    x2: usize,
    y2: usize,
  ) -> (usize, usize, usize) {
    let dx = self.diff(row, x2, x1);
    row += 1;
    let dy = self.diff(row, y2, y1);
    row += 1;
    // λ = dy / dx  ⇒  λ·dx = dy : prover supplies λ, row asserts λ·dx ≡ dy.
    let lam_val = (&self.w[dy] * mod_inv(&self.w[dx], &self.p)) % &self.p;
    let lam = self.alloc(lam_val);
    self.push_row(row, &[(lam, 1)], &[(dx, 1)], &[(dy, 1)]);
    row += 1;
    let t = self.mul(row, lam, lam); // t = λ²
    row += 1;
    let t1 = self.diff(row, t, x1); // t1 = t − x1
    row += 1;
    let x3 = self.diff(row, t1, x2); // x3 = t1 − x2 = λ² − x1 − x2
    row += 1;
    let dx3 = self.diff(row, x1, x3); // dx3 = x1 − x3
    row += 1;
    let u = self.mul(row, lam, dx3); // u = λ·(x1 − x3)
    row += 1;
    let y3 = self.diff(row, u, y1); // y3 = u − y1
    row += 1;
    (x3, y3, row)
  }
}

#[cfg(test)]
mod tests {
  use super::*;

  /// A small generator-ish point on secp256k1 and its double, for testing.
  /// G = standard secp256k1 generator.
  fn g() -> (BigUint, BigUint) {
    let gx = BigUint::parse_bytes(
      b"79BE667EF9DCBBAC55A06295CE870B07029BFCDB2DCE28D959F2815B16F81798",
      16,
    )
    .unwrap();
    let gy = BigUint::parse_bytes(
      b"483ADA7726A3C4655DA4FBFC0E1108A8FD17B448A68554199C47D08FFB10D4B8",
      16,
    )
    .unwrap();
    (gx, gy)
  }

  /// 2G (precomputed), for an add of two distinct points G + 2G = 3G.
  fn g2() -> (BigUint, BigUint) {
    let p = secp256k1_p();
    let (gx, gy) = g();
    // double G via affine doubling: λ = 3x²/(2y)
    let lam =
      (BigUint::from(3u32) * &gx * &gx % &p) * mod_inv(&(BigUint::from(2u32) * &gy % &p), &p) % &p;
    let x2 = (&lam * &lam + &p + &p - &gx - &gx) % &p;
    let y2 = (&lam * (&gx + &p - &x2) % &p + &p - &gy) % &p;
    (x2, y2)
  }

  #[test]
  fn ec_add_gadget_is_satisfied_and_correct() {
    let p = secp256k1_p();
    let (gx, gy) = g();
    let (g2x, g2y) = g2();
    let expected = ref_ec_add(&gx, &gy, &g2x, &g2y, &p); // 3G

    // Reserve 4 input columns + room; const col placed after a power-of-two pad.
    let const_col = 64usize; // generous; > all columns we'll allocate
    let mut cb = CircuitBuilder::new(const_col, p.clone());
    let x1 = cb.alloc(gx);
    let y1 = cb.alloc(gy);
    let x2 = cb.alloc(g2x);
    let y2 = cb.alloc(g2y);

    let (x3, y3, n_rows) = cb.ec_add(0, x1, y1, x2, y2);
    assert_eq!(n_rows, 9, "affine add should be 9 rows");

    // Output matches the reference 3G.
    assert_eq!(cb.w[x3], expected.0, "x3 mismatch");
    assert_eq!(cb.w[y3], expected.1, "y3 mismatch");

    // Self-check the full relation A·z ∘ B·z = C·z + m∘q on every row.
    let z = &cb.w; // z includes the const-1 at const_col
    for row in 0..cb.mods.len() {
      let lc = |entries: &[Triple]| -> BigUint {
        entries
          .iter()
          .filter(|(r, _, _)| *r == row)
          .map(|(_, col, k)| k * &z[*col])
          .sum()
      };
      let az = lc(&cb.a);
      let bz = lc(&cb.b);
      let cz = lc(&cb.c);
      assert_eq!(
        &az * &bz,
        &cz + &cb.mods[row] * &cb.q[row],
        "row {row} unsatisfied"
      );
    }
  }
}
