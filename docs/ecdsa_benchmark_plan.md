# ECDSA-verify benchmark vs Zinc+ — plan

Goal: implement an ECDSA-verification benchmark in integer Mod-R1CS, minimizing
constraints, to compare against Zinc+'s ECDSA benchmark.

## The comparison target (Zinc+)

- **Zinc+** = eprint 2026/855, "SNARKs for Polynomial Rings" (NethermindEth,
  Garreta et al.). Hash/code-based (Brakedown-type commitment + IOP-of-proximity
  to integers), **AIR/UCS** (trace-based), **not** R1CS, **not** Bünz.
- **ECDSA benchmark:** **secp256k1**. In-circuit = **7× SHA-256 compressions +
  the MSM** (`R = u₁·G + u₂·Q` via a 256-round Shamir double-and-add). The
  **F_n scalar work** (`s⁻¹`, `u₁`, `u₂`, final `R_x ≡ r mod n`) is done
  **off-circuit by the verifier**.
- **Trick:** proving field = secp256k1 **base field F_p** → native EC (no
  quotients), **Jacobian** coords, **no in-loop inversions** (one deferred,
  even moved off-protocol).
- **Numbers:** prover **40.6 ms**, verify **7.0 ms**, proof **198 KB**; no-ZK,
  100-bit security, MacBook Air **M4**, SIMD+Rayon. Trace 512 rows, ~13
  constraints deg ≤7. Repo: `github.com/NethermindEth/zinc-plus` (`main-beta`),
  `test-uair/src/ecdsa*.rs`.

## SOTA constraint counts (all digest-in, SHA excluded)

| system | curve | constraints | technique |
|---|---|---|---|
| circom-ecdsa | secp256k1 | **1,508,136** | naive non-native double-and-add |
| efficient-zk-ecdsa | secp256k1 | 163,239 | efficient-ECDSA + precompute (still emulated) |
| gnark | secp256k1 | ~122k R1CS | emulated field + lazy reduction |
| gnark Fake-GLV | P-256 | 195,266 | + Fake-GLV (½ the ladder) |
| **spartan-ecdsa (floor)** | secp256k1 | **3,039** | **secq curve cycle (native) + efficient-ECDSA (1 scalar mult)** |

SHA-256 ≈ 27–30k R1CS if included; almost all benchmarks take the digest as
input. The dominant cost everywhere is the EC scalar mult(s); the inverse is a
1-mult hint (`a·a⁻¹≡1`), never in-circuit Euclid.

## Our structural advantages (why integer Mod-R1CS fits ECDSA)

1. **EC arithmetic mod p is a *modular row*, not field emulation.** Each
   `a·b mod p` is one row `a·b = c + p·q` (per-row modulus = secp256k1's `p`).
   No limbs / per-limb range checks — this is the lever that takes 1.5M → ~k's,
   same effect as spartan-ecdsa's secq cycle but via per-row moduli.
2. **Affine coordinates, because division is 1 row.** Prover supplies the slope
   `λ` as advice; `λ·(x₂−x₁)=y₂−y₁` is one row. So an **affine add ≈ 3 rows**,
   **double ≈ 4 rows** — *cheaper* than Jacobian (~13 mults). This is the
   opposite of Zinc+'s choice (they use Jacobian *because* their AIR can't do
   cheap inversions). **Our cheap division flips the optimal coordinate system.**
3. **Multiple moduli in one proof** → we can prove the **full** verify including
   the F_n block (`s⁻¹`, `u₁`, `u₂`, `R_x ≡ r mod n`) at ~1 row each — the part
   Zinc+ punts to the verifier. A "we prove strictly more" story.
4. **Range checks ~free** — norm bounds handled by the IntEval PCS, not explicit
   bit-decomposition rows.

## Scope (match Zinc+ for apples-to-apples)

- **In-circuit:** the 2-scalar MSM `u₁·G + u₂·Q` (256-round affine Shamir).
- **SHA-256: excluded** (digest-in — matches all SOTA; SHA is bitwise = Zinc+'s
  binary-ring branch, our weak spot ≈27–30k emulated rows; report separately if
  asked, don't compete on it).
- **F_n block:** keep in public to match Zinc+; optionally add in-circuit (~few
  rows) as a "full-verify" variant — our completeness advantage.
- **Curve:** secp256k1; per-row moduli = secp256k1 `p` and `n`. Our PCS
  commitment curve (T256) is independent.

## Negation / non-negativity (soundness-critical)

The system assumes **non-negative witnesses** (`nonnegative-witness` memory).
EC arithmetic needs subtractions, and the obvious `(p−1)`-coefficient negation
(`−x ≡ (p−1)x mod p`) is **UNSOUND** here: it yields a **negative quotient** in
edge cases (e.g. `λ=0` ⇒ product `< LC_C` ⇒ `q<0`). So every subtraction `a−b`
uses a **difference witness**: introduce `d = (a−b) mod p ∈ [0,p)` and one row
`d + b ≡ a (mod p)` (quotient `∈{0,1}`, ≥0). All values stay `< p`, all `q < p`
(256-bit) ⇒ `log_t_f = 256` (same regime as msshape; no widening).

Consequence: an affine add is ~**6–7 rows** (diff rows + the products), not 3;
a double ~**7–8 rows**. Honest MSM estimate revised below.

## Circuit decomposition (affine Shamir MSM, secp256k1, `a=0`)

Public inputs (mirroring Zinc+): `u₁,u₂` and their bits, table `{O, G, Q, G+Q}`,
and `r`/`R`.
1. Precompute `G+Q` = 1 affine add (~3 rows).
2. 256 rounds, accumulator `P`:
   - `P ← 2P` — affine double: `λ=3x²/(2y)` → `λ·2y = 3x²` (needs `x²`, ~2 rows),
     `x₃=λ²−2x` (1 row), `y₃=λ(x−x₃)−y` (1 row). ~4 rows.
   - `P ← P + T` where `T` = table point selected by `(u₁[i],u₂[i])` — selection
     `T = b₁(1−b₂)G + (1−b₁)b₂Q + b₁b₂(G+Q)` inlined (~2 rows), affine add ~3 rows.
3. Final: `R_x ≡ r (mod n)` order check (optional in-circuit, 1 row).
- **Identity / incomplete-addition handling (design point):** affine add is
  singular when the two points are equal or `O`. Mirror Zinc+'s fix — start the
  accumulator at a non-identity **seed** `R_init` and offset the result by
  `R_init^(2²⁵⁶)`, OR prove input distinctness. Must be handled for soundness.

**Estimated size (revised, sound):** ~256 rounds × (double ~8 + add ~7 +
select ~2) ≈ **~4.5–8k rows** → pad to **2¹³**. Same order as the spartan-ecdsa
floor (3,039) and far below circom/gnark; the diff-witness rows are the ~2×
versus the optimistic count. (Efficient-ECDSA reformulation → **one** in-circuit
scalar mult ≈ halves it — optional, diverges from Zinc+'s 2-scalar MSM.)

## Gadgets to build (none exist; hand-wired like multiswap)

- `affine_add(P,Q) mod p` (3 rows) · `affine_double(P) mod p` (4 rows)
- `shamir_select` (table point by 2 bits, ~2 rows)
- `shamir_msm` loop (256 rounds) + seed/offset identity handling
- optional F_n block: `s⁻¹` / `u₁` / `u₂` / `R_x mod n` (1 row each)
- witness gen: real secp256k1 scalar mult (compute every coord + slope) and
  per-row quotients via `div_rem` (template: `multiswap_modp.rs`)
- a known-answer test against a real secp256k1 library (accept valid sig, reject
  tampered)

## Prover-time caveat & the Brakedown coupling

Constraint count is the clean comparison; **prover time is apples-to-oranges**:
Zinc+ = tight Brakedown AIR + SIMD on M4 (40.6 ms *with* SHA, multi-thread); ours
= Mod-PCS. At ~2¹² rows / 256-bit coeffs our **Hyrax** Mod-PCS is ~hundreds of ms
single-thread — *not* competitive, because the commit MSM dominates. **This is
exactly why we built Brakedown:** to be competitive with Zinc+'s Brakedown-based
prover we need our **Mod-PCS over Brakedown** (the deferred ~40–60% rewrite). So:

- **Phase 1 (this task):** ECDSA circuit + current Hyrax Mod-PCS → constraint
  count (the strong story) + a baseline prover time.
- **Phase 2 (later):** Mod-PCS over Brakedown → competitive prover time vs Zinc+.

## Build order

1. affine EC add/double gadgets + KAT vs a secp256k1 lib.
2. Shamir MSM builder + witness gen + `is_sat`.
3. full ECDSA circuit (+ optional F_n) + valid/invalid signature test.
4. bench harness (setup/prove/verify) → constraint count + Hyrax baseline time.
5. (Phase 2) Brakedown integration → competitive numbers.

## Decisions (resolved 2026-06-22)

- **Scope = the MSM only** (user): the 2-scalar `u₁·G + u₂·Q` 256-round affine
  Shamir loop, matching Zinc+'s in-circuit MSM. `u₁,u₂` are given (public inputs,
  bit-decomposed). **No F_n block, no SHA, no full-verify** in this first pass.
- Output is `R = u₁·G + u₂·Q` (optionally the `R_x` value); the `R_x ≡ r mod n`
  check stays out for now.
- Phase-1 with the current **Hyrax** Mod-PCS for a constraint-count + baseline
  number; Brakedown integration is the later phase for competitive prover time.

(Deferred / optional: efficient-ECDSA 1-scalar-mult reformulation, the in-circuit
F_n block — both noted above; not in the first build.)
