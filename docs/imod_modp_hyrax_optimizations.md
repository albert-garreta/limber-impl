# imod_spartan_modp — low-hanging prover/verifier optimizations (Hyrax-kept)

Optimizations that reduce prover/verifier cost of `IntModSpartanModpSNARK`
**without** changing the PCS (stays Hyrax-over-T256) and **without** any
protocol/soundness change. All targets live in the Phase-3 batch
range-check path (`src/provider/pcs/integer_modpcs.rs`), which the bench
notes flag as the 3–4× cost driver — so this is where the leverage is.

Baseline commit: `99fc22f` (batched D5 range checks per `(bound, size)`
group).

## Why the range-check path

`imod_spartan_modp.rs` itself has no range checks (`//! ... no range
checks`, line 26). The soundness-grade range check lives in
`IntegerModPCS` as the step-D5 `rbatchrange` argument
(`prove_batch_range_check` / `verify_batch_range_check`). Each
`(bound, size)` group emits `1 + 2t` batched checks, and each batched
check runs: one Hyrax commit, a bit-validity zerocheck, a
value-reconstruction sumcheck, and 3–4 Hyrax openings. The items below
shrink the per-check constant factors.

## Ranked items

### 1. Bit-validity zerocheck: collapse 3 identical polys → 1  *(highest confidence, smallest diff)*

**Where:** `prove_batch_range_check`,
`src/provider/pcs/integer_modpcs.rs:1996-2007`.

**Problem:** the bit-validity zerocheck calls
`SumcheckProof::prove_cubic_with_three_inputs` with
`poly_A = poly_B = poly_C = bit_poly.clone()`. The integrand that routine
computes is `eq(x,τ)·(A·B − C)` (`sumcheck.rs:1114+`,
`*zero_a * *zero_b - *zero_c`), so with all three equal it evaluates
`eq·(bit² − bit)` — the bit-validity check. But:

- All three polys start identical and are bound by the **same** `r_i`
  every round (`sumcheck.rs:555-562`), so they remain bitwise identical
  for the entire sumcheck.
- Cost paid for nothing: **3× `n_bits` allocations** (two redundant
  clones) and **3 `bind_poly_var_top` calls per round** where one
  suffices (~2/3 of this sumcheck's binding work is wasted).

**Fix:** add a single-input specialization
`SumcheckProof::prove_cubic_square(claim, taus, poly_A, transcript)` that
computes the round polynomial from one poly with integrand
`eq·(A² − A)`:

```
t_0   = a0 * a0 - a0          // = a0² − a0
t_inf = (a1 - a0) * (a1 - a0) // = (a1 − a0)²
```

bind once per round, and return `[poly_A[0]]`. Swap the call site to pass
a single `bit_poly` copy.

**Expected impact:** removes 2 of 3 large allocations and ~2/3 of the
per-round binding cost of the bit-validity sumcheck; soundness-neutral
(identical integrand). Fires once per range-check group.

**Risk:** very low. New code path is a strict specialization of an
existing, tested routine; verify side is unchanged (same transcript
absorbs, same round-poly degree 3).

### 2. Parallelize the bit-decomposition build  *(high confidence, embarrassingly parallel)*

**Where:** `prove_batch_range_check`,
`src/provider/pcs/integer_modpcs.rs:1973-1983`.

**Problem:** `bit_poly` is filled by a sequential nested loop over
`(p, within)`, each iteration writing a disjoint `stride`-sized slice via
`bit_decompose_value`. This is a serial section that grows with the batch
(`N · n_values · log_bound` writes), single-threaded.

**Fix:** parallelize over the disjoint output slices, e.g.
`bit_poly.par_chunks_mut(stride)` indexed by `(p·n_values + within)`, or
a `par_iter` over the flattened `(p, within)` space writing into
non-overlapping ranges. No data races (ranges are disjoint by
construction).

**Expected impact:** removes a serial section on the large configs
(`num_cons = 2^10`). Bounded by core count.

**Risk:** low. Pure data-parallel rewrite of an existing loop; output
bytes identical.

### 3. `stride` zero-padding waste  *(medium effort, biggest latent waste)*

**Where:** `stride = 1 << log_log_bound`,
`src/provider/pcs/integer_modpcs.rs:1966`; consumed throughout the
range-check (`n_bits = n_pad * n_values * stride`).

**Problem:** `stride = 2^⌈log₂ log_bound⌉`. When `log_bound` is not a
power of two (e.g. `log_bound = 33 → stride = 64`), nearly half of
`n_bits` is provably-zero padding that still costs the commit, both
sumchecks, and all the binding/eval work.

**Fix (investigate, then decide):**

- First **measure**: log the realized `(log_bound, stride, n_bits)` per
  group under `RUST_LOG=info` and see how often `stride > log_bound` and
  by how much for the bench's bound distribution.
- If wasteful: pack values more tightly along the b-axis (e.g. multiple
  values per `stride` block) or choose a `stride` closer to `log_bound`,
  keeping the index map `((p·n_values + within)·stride + b)` consistent
  on both prover and verifier.

**Expected impact:** up to ~2× on the range-check when bounds sit just
above a power of two.

**Risk:** medium. Touches the shared prover/verifier index layout — must
keep `verify_batch_range_check` in lockstep. Gate behind measurement
from step 0.

### 4. Batch the per-opening `comm_eval` / IPA overhead  *(medium effort, also helps verifier)*

**Where:** `hyrax_open_at`,
`src/provider/pcs/integer_modpcs.rs:1945-1946`; call sites in
`prove_batch_range_check` (`:2009-2086`).

**Problem:** every `hyrax_open_at` commits a 1-element `comm_eval` and
runs a full IPA. Each `BatchRangeCheck` does 3 bit-opens + 1 value-open,
and there are `1 + 2t` checks per group. Two of the bit-opens are on the
**same** `bit_comm` (at `r_validity` and at `r_v ++ r_b`) — candidates
for a single 2-point / batched opening that shares IPA setup. This is the
highest-payoff **verifier**-side item (fewer IPA verifies + 1-element
commits).

**Fix:** introduce a batched-opening helper for multiple eval points
against one commitment; route the two `bit_comm` opens (and, where
points coincide across checks, value opens) through it.

**Expected impact:** fewer IPA prove/verify rounds and 1-element commits
per group; scales with `t`.

**Risk:** medium. Requires a new opening primitive and matching verifier
logic; verify with the existing range-check unit tests before/after.

## Sequencing

1. **#1** — implement `prove_cubic_square`, swap the call site.
2. **#2** — parallelize the bit-decomposition build.
3. Measure (#1 + #2) before going further.
4. **#3 step 0** — instrument `(log_bound, stride, n_bits)`; decide.
5. **#4** — batched opening, if the opening overhead shows up in spans.

Items #1 and #2 are the genuine low-hanging fruit: small, local,
soundness-preserving, hitting the hottest path. #3 and #4 are larger and
should be gated on measurement.

## Measurement protocol

Per-part timing is already wired into the bench, gated on `RUST_LOG`:

```
RUSTFLAGS="-C target-cpu=native" RUST_LOG=info \
  cargo bench --bench imod_spartan_modp
```

This installs the fmt subscriber and prints the section spans
(`imod_pcs_chain_openings`, `imod_pcs_rc_ab`, the range-check sumcheck
rounds, …) for one setup/prove/verify per config without criterion's
iteration noise. Record the range-check spans before/after each item.

Plain `cargo bench` (no `RUST_LOG`) gives the criterion
prove/verify/setup numbers across the configs
`(2^6,2^8), (2^8,2^10), (2^10,2^12)`.

## Invariants to preserve

- No protocol/soundness change: every item is a constant-factor or
  parallelism rewrite. The transcript layout, round-poly degrees, and
  emitted proof contents must be unchanged (except #3/#4, which change
  layout/opening structure on **both** sides in lockstep).
- Keep Hyrax-over-T256 as the underlying PCS; no curve change.
- `cargo clippy` clean — note the repo's `is_multiple_of` requirement
  (do not swap for `%`).
