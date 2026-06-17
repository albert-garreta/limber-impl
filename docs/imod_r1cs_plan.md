# Integer Mod-R1CS on top of Spartan — implementation plan

Tracks the protocol from *SNARKs for Integers* (Def 5.4, `rimodrocslimb`)
ported onto this Spartan codebase. Research prototype; aiming for
concretely good benchmarks. PCS swap to IntEval is later work.

## Goal

A working prover/verifier for the **limb-split Integer Mod-R1CS**
relation, structured so that we can later (a) swap the final PCS opening
for the IntEval protocol and (b) generalize to dual fields (sumcheck
field `p` ≠ PCS field `q`).

## Relation recap (informal)

```
A z ∘ B z = C z + m ∘ q              -- over Z, with bounded norms
```

Limb-split:

```
(A ⊗ ℓ_z) z_limb ∘ (B ⊗ ℓ_z) z_limb
    = (C ⊗ ℓ_z) z_limb + m ∘ (I ⊗ ℓ_q · q_limb)
||z_limb||, ||q_limb||, ||A||, ||B||, ||C|| < B,   ||m|| < B_m
```

Sumcheck (mod random prime `p`):

```
Σ_i eq(i, α) · ( a(i)·b(i) − c(i) − m(i)·q(i) )  ≡_p  0

with virtual polys
  a(i) = Σ_{j,k} A(i,j) · ℓ_z(k) · z_limb(j,k)   mod p
  ... similarly b, c
  q(i) = Σ_k ℓ_q(k) · q_limb(i,k)                mod p
```

Inner step: prove the openings of `a, b, c, q` at the random
challenge `r_i`. Paper splits this into (i) a batched sumcheck over A,
B, C against `z_limb` of length `n + ν_z`, and (ii) a small sumcheck
over `q_limb` of length `ν_q`.

## Scope of phase 1 (single field, no limb-split)

Pin `p = q` (the curve scalar field). Set `ν_z = ν_q = 1` so
`z_limb = z`, `q_limb = q`. This exercises every structural change
*except* the dual-field bridging and the limb folding inside the
sumchecks.

Target test: a hand-built tiny Int-Mod-R1CS encoding
`a · b ≡ c  (mod N)` for some prime `N ≠ Fq`. The witness includes the
quotient `k = (a·b − c)/N`. Single row, mods vector `[N]`.

## Code layout (additive — no edits to existing files)

```
src/
  imod_r1cs/
    mod.rs             -- IntModR1CSShape, IntModR1CSWitness, IntModR1CSInstance
    sparse.rs          -- (later) reuse src/r1cs/sparse.rs as-is for now
  imod_spartan.rs       -- IntModSpartanSNARK driver (clone of spartan.rs, modified)
  sumcheck.rs           -- add prove_cubic_with_four_inputs (extension, not edit)
```

`lib.rs` gets two new `mod` lines. Existing tests untouched.

## Type sketches

```rust
pub struct IntModR1CSShape<E: Engine> {
    num_cons: usize,
    num_vars: usize,         // |w|
    num_io:   usize,         // |x|
    A: SparseMatrix<E::Scalar>,
    B: SparseMatrix<E::Scalar>,
    C: SparseMatrix<E::Scalar>,
    mods: Vec<E::Scalar>,    // length num_cons
    // norm bounds carried as metadata (B, B_m, B_z, B_q); not enforced phase 1
}

pub struct IntModR1CSWitness<E: Engine> {
    w: Vec<E::Scalar>,       // |w| = num_vars
    q: Vec<E::Scalar>,       // |q| = num_cons (one quotient per constraint row)
    r_w: Blind<E>,
    r_q: Blind<E>,
}

pub struct IntModR1CSInstance<E: Engine> {
    comm_w: Commitment<E>,
    comm_q: Commitment<E>,
    x: Vec<E::Scalar>,
}
```

## Protocol changes vs `spartan.rs`

| Step | Spartan today | Phase 1 IntMod |
|---|---|---|
| Commit | `comm_W` only | `comm_W`, `comm_Q` |
| Outer SC integrand | `eq · (Az·Bz − Cz)` | `eq · (Az·Bz − Cz − M·Q)` with M, Q tracked separately |
| Outer SC degree | 3 | 3 (eq=1, AB=2, MQ=2 → cubic) |
| Outer SC inputs | 3 vectors (Az, Bz, Cz) | 5 vectors (Az, Bz, Cz, M, Q). New helper `prove_cubic_with_five_inputs` |
| Outer claims | `(v_a, v_b, v_c)` | `(v_a, v_b, v_c, v_m, v_q)`. Final check: `eq(r_x,α) · (v_a·v_b − v_c − v_m·v_q)` matches |

**Why 5 inputs and not 4.** Treating the m∘q term as a single committed
polynomial doesn't work: the MLE of pointwise-product ≠ pointwise-product
of MLEs, so the verifier can't reconstruct the combined eval from the
separate openings of m and q. The fix is to keep M and Q as separate
multilinears throughout the SC; the round polynomial gains a quadratic
contribution from `M·Q` but the total degree stays 3 (eq still caps it).
BDDT formulas:
- `t_0 = a_0·b_0 − c_0 − m_0·q_0`
- `t_inf = (a_1−a_0)(b_1−b_0) − (m_1−m_0)(q_1−q_0)`
| Inner SC | one quad SC over `(A+rB+r²C)z` against `z` | unchanged for w; add a tiny extra SC opening for `q` at `r_x` |
| Final eval | one PCS open for `w` | two PCS opens (`w`, `q`) |

The "one tiny extra SC" for `q`: at the end of outer SC the verifier
needs `q̅ = q(r_x)`. Since `q` is committed and has `log num_cons`
variables, this is just an evaluation claim — fold it into the PCS
opening, no extra sumcheck. (Paper's structure has a sumcheck because
of the `ℓ_q` limb fold; with `ν_q = 1` that fold is trivial.)

## What we explicitly defer

- **Range checks** on `w`, `q`. Prover-side debug assertion only in
  phase 1. Real PIOP in phase 3.
- **Limb splitting** (`ν_z, ν_q > 1`). Phase 2.
- **Dual fields** (`p ≠ q`). Phase 4. We use the curve scalar field as
  both. The soundness analysis from the paper still holds as long as
  the curve scalar field is ≥ `λ` bits.
- **Sparse matrix evaluation as a subprotocol** (`rsparseeval`). Reuse
  `evaluate_with_tables_fast` from `SplitR1CSShape`. OK for `n` up to
  ~24; revisit for bigger.
- **IntEval protocol**. Phase 5.
- **Frontend (bellpepper integration)**. Phase 1 hand-builds the matrices
  in the test.
- **Coprocessor model.**

## Open questions for later phases

- Does the existing `SplitR1CSShape` (with its `shared`/`precommitted`/
  `rest` witness partitioning and the manual round-0 BDDT optimization)
  carry over usefully, or do we want a flatter shape first? Phase 1 will
  start flat (single witness segment); revisit before phase 4.
- Concrete choice of `B` (limb size). Paper suggests `2^32` or `2^64`;
  affects PCS char `q` lower bound. Decide alongside phase 2.
- Where `m` lives — `(idx, x)` says it's part of the index, and the
  verifier evaluates it directly. For phase 1 we keep `m` as an explicit
  field of the shape (preprocessed, public).

## Risk / feasibility summary

Low risk: phases 1–2 are structurally Spartan with bookkeeping. The
sumcheck protocol itself doesn't need to change except for the
4-vector outer round.

Medium risk: phase 4 (dual fields) touches every file that mentions
`E::Scalar` inside sumcheck. Worth holding off until 1–3 are solid.

The IntEval protocol (phase 5) is the genuinely novel piece and the one
most worth landing carefully — but by the time we get there, we'll
have a stable PIOP to plug it into.

## Commitment cost note

`num_cons ≈ num_vars` is typical, so `comm_Q` roughly doubles the witness
commitment work vs. plain Spartan. Mitigations:

- `q` is small-norm by construction (`||q|| < B_q`, e.g. `2^32`–`2^64`).
  Use the existing `is_small` MSM fast path — ~4–8× per-element speedup
  vs. general field elements.
- Phase 2: `z_limb` is also small-norm, so both commitments go through
  the fast path. Total committed bits become competitive with plain
  Spartan-with-emulated-non-native (the relevant baseline).
- If the PCS supports it, commit `[z_limb || q_limb]` jointly to share
  MSM setup.

Benchmark framing: the right baseline is *plain Spartan emulating
non-native arithmetic* (which inflates `num_vars` by a constant per
modular op), not plain Spartan over native field — otherwise the 2×
commitment looks like a regression instead of a win.
