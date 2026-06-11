# GKR leaf-layer univariate skip — design note

Next optimization for the LogUp-GKR range check (`src/logup_gkr.rs`),
following the eq-factoring (`0e218ee`) and all-ones-numerator
(`249276c`) prover optimizations. Unlike those, this one **changes the
proof format and verifier** — it is not transcript-compatible.

## Motivation

The leaf layer of each fraction tree is the largest sumcheck (its round
evals are ~half of all GKR sumcheck work; ~120–150 ms of the ~1.25 s
MultiSwap prove, ~0.5 s single-threaded). Its **first round** operates on
fully structured data, because no challenge has been bound yet:

- witness trees: numerators ≡ 1 (already specialized away); denominators
  `b[i] = r + w[i]` with `w[i] < 2^16` — so diffs `b1−b0 = w1−w0` are
  ~17-bit signed ints and products decompose as
  `b0·b1 = r² + r·(w0+w1) + w0·w1` with small coefficients;
- table tree: denominators `r + j` (index-affine), numerators are the
  multiplicities (small counts).

After round 0 binds its challenge, all tables become full-width random
field elements and nothing further is special. A **univariate skip** of
the first `ℓ` variables extracts `ℓ` rounds of progress while the data is
still small: treat `x_{0..ℓ}` as one variable `v` over an extended domain
`D` (|D| = 2^ℓ points beyond the boolean cube, degree ≈ 3·2^ℓ − 1 for the
cubic integrand), send the round polynomial in evaluation form, and bind
all ℓ variables at once with one Lagrange fold.

## Why the naive version saves nothing

`t256::Scalar::from(u64)` costs one Montgomery multiplication, so lifting
small values per index eats the savings (≈6 → 5 full mults per index).
The win requires keeping the skip-round accumulation in **integer space**:

1. per index, compute `w0+w1`, `w0·w1`, `d0·d1` etc. as u64/i128;
2. accumulate `Σ eq_i · small_i` with a **one-limb Montgomery multiply**
   (4-limb field element × ≤64-bit scalar, ~4× cheaper than full 4×4),
   plus `DelayedReduction`-style unreduced accumulators;
3. exploit `Σ_i eq_i = 1` (partition of unity) to fold the constant terms
   (`r`, `r²`, λ-multiples) out of the per-index loop entirely.

**Prerequisite:** expose a `mul_small(&self, u64) -> Self` (one-limb
Montgomery mult) on `t256::Scalar` / `MontgomeryLimbs`, with a signed
variant for diffs. The MSM layer (`msm_small`) already does the
equivalent internally; this lifts it to a scalar primitive.

## Sketch

Leaf layer of a tree with `2^k`-size half-tables, skip width `ℓ` (e.g. 4):

- Round-0′ (the skip round): for each of ~`3·2^ℓ` evaluation points `v`
  of the degree-(3·2^ℓ−1) round polynomial `s(v)`, accumulate over
  `2^{k−ℓ}` indices using the small-value decomposition. The per-point
  extrapolated table values stay small (linear combinations of `w`'s with
  small Lagrange-ish integer coefficients over `D` — choose `D =
  {0,1,…,2^ℓ·deg}` so extrapolation coefficients are small integers).
- Verifier: receives `s` in evaluation form (`3·2^ℓ` values), checks
  `Σ_{v∈cube part} s(v) = claim` analog for the skip domain, evaluates
  `s(r_v)` by barycentric interpolation, and the eq-factor contribution
  `E(r_v)` becomes a product of ℓ linear factors (Gruen factoring
  generalizes; precompute the prefix product).
- After the skip: one Lagrange fold binds `2^ℓ → 1` (cost `2^k` full
  mults, same as ℓ sequential binds), then the remaining `k−ℓ` rounds run
  the existing factored path on full-width data.

Soundness: degree grows to ~`3·2^ℓ`; Schwartz–Zippel loss `3·2^ℓ/|F|` per
skip round — negligible over the 256-bit field for any practical ℓ.

## Expected gains

Rounds `1..ℓ` of the leaf layer currently cost ~`2^{k−1}` full-mult
index-evals total; the skip converts them (plus round 0) to small-int
work. Estimated: GKR sumcheck −30–40%, end-to-end prove ~1.25 → ~1.15 s
multithreaded, single-threaded ~4.5 → ~4.1 s. Verifier: +`3·2^ℓ` field
elements per tree in proof size, small constant extra verify work.

Given the modest end-to-end delta, do this only after (or together with)
the bigger structural item: `log_t = 16` limbs, which would make the
committed limb polynomials coincide with the 16-bit chunk polynomials,
deleting the separate chunk commitments and roughly halving GKR leaves —
that change *increases* the relative value of this one (the leaf layer
becomes a larger fraction of what remains).

## Implementation steps

1. `mul_small` / `mul_small_signed` scalar primitives + tests
   (benchmark: should be ≥3× faster than full mult).
2. Skip-round prover for the witness-tree leaf layer (a-tables implicit,
   b-tables `r + w`), behind a `ones_numerator`-style flag; keep the
   table tree on the existing path initially.
3. Proof-format change: `GkrLayerProof` gains an optional evaluation-form
   skip round; verifier barycentric evaluation + generalized `E` factor.
4. Differential test: skip prover vs current prover must produce the
   same reduced claims (different transcripts, same final point/eval
   distribution); roundtrip + tamper tests.
5. Re-tune ℓ on the MultiSwap bench (expect ℓ ∈ {3,4,5}).
