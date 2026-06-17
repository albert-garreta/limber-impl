# Benchmarking MultiSwap on `imod_spartan_modp` — implementation plan

Plan for a new bench, `benches/multiswap_modp.rs`, that measures the
prover/verifier cost of proving **MultiSwap** (Ozdemir, Wahby, Whitehat,
Boneh, *Scaling Verifiable Computation Using Efficient Set Accumulators*,
USENIX Security 2020, §3–4) with `IntModSpartanModpSNARK`.

Decisions locked in:

- **Fidelity:** real-arithmetic core. Real big-int group exponentiations
  mod a real `N`, real reductions mod a real prime `ℓ`, correct
  quotients — so the Phase-3 D5 range checks run at true ~2048-bit width.
  The hashes (`H`, `Hp`, `H∆`) are modeled by operation count, clearly
  labeled, not faithful crypto circuits.
- **Operand width:** real 2048-bit, via a small additive
  `setup_with_params` hook (the default setup caps values at 32 bits).

## Background: what MultiSwap costs, and why `imod_spartan_modp` fits

MultiSwap verifies a batch of `k` swaps against an RSA accumulator by
checking **two Wesolowski proofs** that share one Fiat-Shamir prime
challenge `ℓ`:

- Insertion: `Q_ins^ℓ · ⟦S⟧^(∏ᵢ H∆(yᵢ) mod ℓ) = ⟦S'⟧` in `G = (ℤ/N)*/{±1}`
- Removal: symmetric, with `{xᵢ}` and `⟦S'⟧ → ⟦S⟧`

Cost is dominated by **multiprecision modular arithmetic** (paper Fig. 3):
4 group exponentiations with `|ℓ| = 352`-bit exponents mod a `b_N = 2048`-bit
modulus, 2 group mults, the hash-to-prime `Hp`, and per-swap `∏ H∆ mod ℓ`.
In the paper's xJsnark / F_p representation each `mod N` multiply explodes
into limbs + bit-split range checks (`c_eG(b) = 7044·b` constraints per
exponent), for **~11M R1CS constraints total**.

The `imod_spartan_modp` relation is `A·z ∘ B·z = C·z + m∘q` over ℤ, where
**each row carries its own modulus `mᵢ`** and a prover quotient `qᵢ` — one
row == one modular multiply `LC_A·LC_B ≡ LC_C (mod mᵢ)`
([src/imod_r1cs_modp/mod.rs](../src/imod_r1cs_modp/mod.rs)). A `mod N`
multiply that costs ~7044·(2048/352) constraints in the paper is **one imod
row**. Benchmarking MultiSwap here is precisely the demonstration that the
integer-mod representation collapses non-native arithmetic. This matches
the framing in [docs/imod_r1cs_plan.md](imod_r1cs_plan.md): the honest
baseline is *plain Spartan emulating non-native arithmetic*, not
native-field Spartan.

## MultiSwap → IntMod-R1CS row model

One imod row per modular multiply, mixing moduli within a single shape
(the `mods` vector is per-row). For batch size `k`, using square-and-multiply
(~1.5·b multiplies per b-bit exponent):

| Component | rows | modulus `mᵢ` |
|---|---|---|
| `⟦S⟧^e_ins`, `Q_ins^ℓ`, `⟦S'⟧^e_rm`, `Q_rm^ℓ` (4 exps, ~1.5·352 each) | ~4·528 ≈ 2112 | `N` (~2048-bit) |
| 2 group mults `×G` | 2 | `N` |
| `Hp` hash-to-prime (Pocklington: 4 exps mod pᵢ + Miller-Rabin base) | ~600 | `p₀…p₄` |
| `∏ H∆ mod ℓ`, insert + remove | 2k | `ℓ` (~352-bit) |
| reduce `∆ mod ℓ` once | 1 | `ℓ` |
| base hash `H` per element (×2k, modeled as field arithmetic) | 2k·`H_ROWS` | `p_hash` (BLS12-381-like) |

Total ≈ `2700 + 2k·(H_ROWS+1)`, padded to a power of two for `num_cons`.
`num_vars` is the next pow2 ≥ column count (chained square-and-multiply
reuses columns, so it grows ~linearly with rows, not quadratically). The
accumulator/set size `2^m` does **not** affect row count (it only drives
prover witness-gen time for the real digest, which we do not compute), so
it is **not** a bench axis.

## Implementation steps

### Step 1 — params hook (load-bearing dependency)

`IntModSpartanModpSNARK::setup` → `shape.commitment_key()` hardcodes the
default IntEval params with `DEFAULT_LOG_T_F = 32`
([src/provider/pcs/integer_modpcs.rs:881](../src/provider/pcs/integer_modpcs.rs#L881)),
admitting only ≤32-bit witness values — which is why the existing bench
uses `modulus = 7`. MultiSwap's `mod N` values are ~2048-bit. Add two
additive functions (no edits to existing call paths):

- `IntModR1CSShapeModp::commitment_key_with_params(&self, params: IntEvalParams)`
  in [src/imod_r1cs_modp/mod.rs](../src/imod_r1cs_modp/mod.rs) — mirror
  `commitment_key()` but call `IntegerModPCS::setup_with_params`.
- `IntModSpartanModpSNARK::setup_with_params(shape, params)` in
  [src/imod_spartan_modp.rs:134](../src/imod_spartan_modp.rs#L134) — mirror
  `setup`, routing through the new shape method.

Choose `log_t_f ≥ b_N` (~2048) and a limb bound `log_t` (e.g. 32 →
`numlimb ≈ 64`, the `numlimb > 1` path that exercises the real range-check
cost). `IntEvalParams::derive` validates the soundness bounds and errors
loudly if the `(log_t_f, log_t, k, num_vars)` combination is infeasible —
see the next section.

### Step 2 — `multiswap_shape_and_witness(k, params) -> (shape, w, q, x)`

Emit the rows above with **real big integers**:

- A real 2048-bit `N` (RSA-2048 challenge number from the paper's
  Appendix B is the natural pick) and a real ~352-bit prime `ℓ`.
- Each of the 4 exponentiation chains as real square-and-multiply over ℤ:
  every step is a row `aᵢ·bᵢ = cᵢ + N·qᵢ` with `cᵢ = aᵢ·bᵢ mod N`,
  `qᵢ = (aᵢ·bᵢ − cᵢ)/N`. Chain outputs feed the next step's column.
- `∏ H∆ mod ℓ` rows with modulus `ℓ`; `Hp`/Pocklington exps with small
  fixed primes `p₀…p₄`; per-element base-hash blocks as `a·b = c + p_hash·q`
  rows (modulus = BLS12-381-like prime), `H_ROWS` rows per invocation (a
  named const documented as a model).
- Debug-assert `shape.is_sat(ck, U, W)` so every config is a valid
  instance before timing.

### Step 3 — Criterion harness

Clone the structure of
[benches/imod_spartan_modp.rs](../benches/imod_spartan_modp.rs):

- Configs `k ∈ {16, 64, 256, 1024}` → `num_cons` ≈ 2^13…2^18.
- `RUST_LOG=info` per-part span dump (reuse the existing gating block) so
  the D5 range-check spans at `numlimb ≈ 64` are visible.
- `setup` / `prove` / `verify` groups via `iter_batched`, plus a `println!`
  of imod row count + proof size per config.
- Register `[[bench]] name = "multiswap_modp", harness = false` in
  [Cargo.toml](../Cargo.toml).

### Step 4 — Reporting / comparison

Per `k`, print imod `num_cons`, prove ms, verify ms, proof bytes — next to
the paper's analytical F_p constraint count at the same `k`
(`2(c_He+c_Hin+c_split+c_+ℓ+c_×ℓ)·k + 4c_eG(352)+2c_×G+c_Hp+…` from Fig. 3,
≈11M for large `k`). That ratio is the headline. The existing
`spartan_synthetic` bench gives a matched-`num_cons` plain-Spartan
wall-clock if a same-size baseline is wanted.

## Why the IntEval bounds can reject a `(width, size)` combination

> Correcting an imprecise claim from the original plan: the risk is **large
> `num_vars`** (a big circuit), *not* small `num_vars`. Small `num_vars`
> actually makes the bounds easier.

The committed integer polynomials live in a PCS field of **fixed** width
`log_q = 256` bits (T256). IntEval splits each wide coefficient into
`numlimb = ⌈log_t_f / log_t⌉` limbs (each `< 2^log_t`) and proves a
random-fold / partial-evaluation over a prime `P` of `log_p` bits. Two
soundness bounds pin `log_p` from opposite directions
([src/provider/pcs/integer_modpcs.rs:243-285](../src/provider/pcs/integer_modpcs.rs#L243-L285)):

- **Partial-Eval Norm Bound** (no wraparound mod `q`):
  `k + k·log_p + max(log_t, log_p) < log_q = 256`.
  This **upper-bounds** `log_p`. A larger limb width `log_t` (which is how
  you keep `numlimb` small for wide values) pushes the ceiling **down**; a
  larger batch `k` does too.
- **Soundness Bound 1** (fold-collision probability ≤ `2^−λ`):
  `s · (log_p − 5 − log₂λ − log₂n) ≥ λ`, where
  `n = num_vars + numlimb_var` is the limb-split polynomial's variable
  count. For any positive `s` to exist you need
  `log_p > 5 + log₂λ + log₂n`. This **lower-bounds** `log_p`, and the floor
  **rises** with circuit size `num_vars` (via `log₂n`).

So `log_p` must satisfy, roughly:

```
5 + log₂λ + log₂n   <   log_p   <   (256 − k − max(log_t, log_p)) / k
```

`log_t_f` itself enters only through `numlimb_var = ⌈log₂ numlimb⌉` — a
small additive bump to `n` (e.g. `log_t_f = 2048, log_t = 32 → numlimb = 64
→ numlimb_var = 6`), so widening operands barely moves the floor directly.
What actually closes the window is pushing **both** ends at once: wide
operands handled with **large limbs** `log_t` (ceiling drops) **and** a
**large circuit** `num_vars` (floor rises). When the window is empty,
`IntEvalParams::derive` returns either *"no log P > 1 satisfies Partial Eval
Norm"* or *"Soundness 1 denominator non-positive"*.

Practical consequence for this bench: prefer **more limbs (smaller
`log_t`)** to represent the 2048-bit values rather than fewer-but-wider
limbs — it keeps the norm-bound ceiling high at the cost of more
range-check work (the very cost we want to measure). Smoke-test the params
at the **largest** config (`k = 1024`, biggest `num_vars`) first, since
that is where the floor is highest and the window most likely to close.

## Validation & risks

- **Correctness gate:** `shape.is_sat` + one full
  `setup_with_params → prove → verify` roundtrip per config inside the
  `RUST_LOG` block (same pattern as the current bench) before criterion
  timing.
- **Soundness-bound risk:** see the section above — validate params at the
  largest config first; if the window is closed, lower `log_t` (more
  limbs) or reduce `num_vars`.
- **Clippy/fmt:** keep `is_multiple_of` (repo requirement — do not swap for
  `%`); `cargo fmt` / `clippy` clean.
- **Explicitly modeled, not faithful crypto** (state in the file header):
  Poseidon / Pocklington / division-intractable hashes and RSA group
  structure are represented by operation count, not real circuits. The
  arithmetic *core* (exps mod `N`, reductions mod `ℓ`, quotients, range
  checks) is real.