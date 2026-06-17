# imod_spartan_modp — low-hanging prover/verifier optimizations (Hyrax-kept)

Optimizations that reduce prover/verifier cost of `IntModSpartanModpSNARK`
**without** changing the PCS (stays Hyrax-over-T256) and **without** any
protocol/soundness change. All targets live in the Phase-3 batch
range-check path (`src/provider/pcs/integer_modpcs.rs`), which the bench
notes flag as the 3–4× cost driver — so this is where the leverage is.

Baseline commit: `99fc22f` (batched D5 range checks per `(bound, size)`
group).

**Status (2026-06-04):** items **#1** and **#2** are implemented and merged
(`prove_cubic_square` + parallel bit-decomposition build); both are
soundness-neutral and verified by the existing 159 tests + clippy. Items
**#3** (stride padding) and **#4** (batched IPA opens) remain open and are
gated on measurement. Separately, the *chain-opening* path (outside this
doc's range-check scope) has since been batched via same-point RLC folds
(`curr_batch`) and same-commitment multi-point sumchecks (`aprev_batch`);
see [imod_followups.md](imod_followups.md) for that line of work.

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

### 1. Bit-validity zerocheck: collapse 3 identical polys → 1  ✅ **DONE**  *(highest confidence, smallest diff)*

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

**Outcome:** implemented as `SumcheckProof::prove_cubic_square` in
`src/sumcheck.rs` (reuses `evaluation_points_cubic_with_three_inputs`
with `A=B=C`, binds once); call site in `prove_batch_range_check` now
passes a single `bit_poly`.

### 2. Parallelize the bit-decomposition build  ✅ **DONE**  *(high confidence, embarrassingly parallel)*

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

**Outcome:** the bit-decomposition build now writes disjoint slices in
parallel; output bytes identical, tests green.

### 3. `stride` zero-padding waste  ⏳ **OPEN**  *(medium effort, biggest latent waste)*

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

### 4. Batch the per-opening `comm_eval` / IPA overhead  ⏳ **OPEN**  *(medium effort, also helps verifier)*

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

1. ~~**#1** — implement `prove_cubic_square`, swap the call site.~~ ✅ done
2. ~~**#2** — parallelize the bit-decomposition build.~~ ✅ done
3. ~~Measure (#1 + #2) before going further.~~ ✅ done
4. **#3 step 0** — instrument `(log_bound, stride, n_bits)`; decide. ← next
5. **#4** — batched opening, if the opening overhead shows up in spans.

Items #1 and #2 (the genuine low-hanging fruit: small, local,
soundness-preserving, hitting the hottest path) are done. #3 and #4 are
larger and gated on measurement. Note that the range-check path still
dominates prove (~63%), so the bigger remaining lever is **follow-on A**
(batching the `1 + 2t` range-check groups into one stacked bit commitment
+ batched opens) rather than the remaining constant-factor items here.

## Follow-ons that change the emitted proof

The items #1–#4 above are constant-factor / parallelism rewrites that
leave the emitted proof byte-identical. The two threads below differ on
exactly one axis: they **add new sub-arguments, so the proof changes**.
They are *not* otherwise alike in size or risk — that distinction lives in
each item's **Risk** line:

- **Thread A** (range-check stacking) is the large, soundness-sensitive
  one: it alters a load-bearing sub-protocol and needs a soundness
  re-check.
- **Thread B** (final-open batch) is small, localized, and **sound by
  construction**: it sends no new commitment (the verifier reconstructs
  the stacked commitment from already-bound per-chain commitments) and
  reuses the exact multi-point-batch pattern of the existing, tested
  `aprev_batch` — purely additive, no cascade. It sits here only because
  it changes the proof bytes, not because it carries A's risk.

Both are the remaining high-value levers, but neither is "no-change to the
emitted proof" in the strict sense of
#1–#4 (they do change the emitted proof).

### A. Batch all bit commitments + openings into one range-check argument  ⏳ **OPEN — biggest lever**

**Where:** `prove_batch_range_check` / `verify_batch_range_check` and
their call sites (`src/provider/pcs/integer_modpcs.rs:1504-1584`).

**Problem:** the range-check path is still ~63% of prove. Keep the
bit-decomposition construction — it works and is sound — but stop paying
for it `1 + 2t` separate times. Prove currently issues **`1 + 2t`
independent `BatchRangeCheck` groups** (one `f_limb` group, plus per
iteration `j` one `a_j` group and one `b_j` group; each group already
batches the `s` chains internally). Every group has its **own**
`bit_comm` Hyrax commitment, its **own** bit-validity zerocheck +
value-reconstruction sumcheck, and its **own** 3–4 Hyrax opens. So the
prover pays `1 + 2t` bit commits and `~4(1 + 2t)` opens, plus `1 + 2t`
sumcheck pairs — most of which are fixed per-invocation overhead
(1-element `comm_eval` commits, IPA setup, sub-transcript spawns).

**Idea:** collapse the `1 + 2t` groups into **one** combined range-check
argument over the concatenation of all range-checked values
(`f_limb ++ all a_j ++ all b_j`):

- **One stacked bit commitment.** Concatenate every group's bit layout
  into a single `bit_poly` and commit once. The groups have different
  `log_bound`s (`f_limb ~ log_t≈32`, `a_j ~ log_p+1`, `b_j ~ 256−log_p+1`)
  and therefore different per-value strides, so the stacked layout carries
  a per-segment `(offset, stride, n_values)` descriptor; bits of segment
  `g` live at `base_g + (within·stride_g + b)`. One `Hyrax::commit`
  replaces `1 + 2t`.
- **One bit-validity zerocheck.** `eq·(bit² − bit)` over the whole stacked
  `bit_poly` (already a `prove_cubic_square` after item #1) — bit validity
  is segment-agnostic, so the union is checked in a single sumcheck.
- **One value-reconstruction sumcheck.** Each segment reconstructs its
  value as `Σ_b 2^b·bit`; combine the `1 + 2t` per-segment reconstruction
  claims by an RLC (`Σ_g μ^g·claim_g`) into a single sumcheck over the
  stacked layout, with the per-segment weight vector `2^b` selected by the
  segment descriptor.
- **Batched opens.** The two opens that hit the same `bit_comm` (at
  `r_validity` and at `r_v ++ r_b`) fold into the stacked commitment; the
  per-segment value opens, where points coincide, fold by RLC (this
  subsumes item #4). Net: `O(1)` bit commit + `O(1)` opens instead of
  `O(1 + 2t)`.

This is exactly the paper's `rbatchrange` stacking (`main.tex` §4), and it
recovers most of the D5 `1 + 2t` blow-up while staying entirely within the
existing bit-decomposition + Hyrax machinery — no new PCS, no lookup
argument.

**Heaviest design item:** the shared stacked bit-layout — either pad every
segment to a common stride or carry per-segment `(offset, stride)`
descriptors through both prover and verifier in lockstep. The
value-reconstruction weight vector must be segment-aware.

**Status update (post-F_a/F_b):** after the `F_a`/`F_b` stacking refactor,
the a/b range groups already collapsed to two (`F_a`, `F_b`), so there are
now **3** groups total (`f_limb`, `F_a`, `F_b`) regardless of `t`. Thread A
is now "merge those 3 into 1".

**Soundness: NOT soundness-sensitive.** This is the *same* `rbatchrange`
construction already shipped and tested (the existing batched range check
stacks N polys into one bit-poly with one commitment + masked `2^b`
weights). Thread A only makes the stacked segments *heterogeneous*
(different `(bound, stride, size)`). There is **no new soundness argument,
no new primitive, no moved/dropped binding**: bit-validity (`bit²−bit=0`)
is segment-agnostic; the value-reconstruction RLC is standard sumcheck
batching; the per-segment masked weights that enforce each bound are
unchanged; and the value bindings stay **explicit** (one open per segment
against its own commitment). A layout/offset bug manifests as a
*completeness* failure (honest roundtrips stop verifying — caught
immediately), and a weight-mask bug is exactly what the existing
`verify_rejects_tampered_range_check` test guards (extended to the merged
group). The *only* variant that would carry a "silent" soundness surface
is folding the value bindings into the multi-point batches as deferred
`V(r_v)` scalars — so we **don't** do that; value opens remain explicit.

**Risk:** low — a contained refactor of `prove_batch_range_check` /
`verify_batch_range_check` (+ call sites), guarded by the existing
roundtrip + range-tamper tests (with a per-segment range-violation case
added). No plan-mode ceremony needed.

**Perf caveat (measured):** the win is the *fixed per-group overhead*
(3 bit commits → 1, fewer bit opens), **not** the bulk — stacking does not
reduce total bits, and the dominant cost is the degree-3 bit-validity +
reconstruction sumchecks, which are proportional to total bits either way.
Worse, the 3 segment bit-counts here sum to a non-power-of-two
(`131072 + 16384 + 131072 = 278528 = 2^18 + 2^14`), so a *single* unified
sumcheck over the union pads to `2^19` — a ~2× blow-up on the expensive
degree-3 zerocheck — while keeping per-segment sumchecks but batching the
6 bit-opens needs a multi-point `W`-build over the large (~`2^19`) bit-poly
that costs about as much as the opens it removes. So a naive full merge is
net-neutral-to-negative at these sizes; the clean sub-win is to **fold the
3 value opens into the existing `f_a`/`f_b`/`aprev` batches** (each
`V_g(r_v)` is an eval of the very commitment that batch already opens),
removing 3 hiding opens with negligible added cost.

### B. Batch the `s` final chain opens via a free-concat stack  ⏳ **OPEN — ~16% prove (bench-size-dependent)**

**Where:** the `final_open` path in `IntegerModPCS::prove` / `verify`
(`integer_modpcs.rs:1387-1419` prover, `:1818-1856` verifier). For the
bench configs (all `t=1`) the `imod_pcs_chain_openings` span is *purely*
these `s` final opens.

**Problem:** the `s` `final` opens (`a_t_c` at `r_i_c`) can join neither
`curr_batch` (same-point RLC fold) nor `aprev_batch` (same-commitment
multi-point sumcheck): each is a **distinct** commitment at a **distinct**
point. Measured at ~16% of prove (`c2^10_v2^12`: 49 ms of 314 ms).

**Idea (and why it is *not* soundness-critical):** stack the `s` final
polys into one polynomial `F` whose commitment is
`combine_commitments([comm_a_t_c]_c)` — the row-concatenation of the
**already-absorbed** per-chain commitments. Crucially the prover sends
**no new commitment**: the verifier *reconstructs* `F`'s commitment from
commitments it has already bound, so there is nothing to cheat with and
**no consistency proof is needed**. Then run the standard multi-point
batch — the *exact* pattern `aprev_batch` already uses and which we trust:

- `W = Σ_c λ^c·eq(POINT_c, ·)`, `claim = Σ_c λ^c·y_c`, where `POINT_c`
  places chain `c`'s point `r_i_c` into `c`'s block of the stacked index
  space (`chain_bits(c) ++ 0… ++ r_i_c`, MSB-first per `EqPolynomial`).
- one degree-2 sumcheck on `F·W`, then **one** open of `F` at the
  sumcheck challenge `r`; verifier checks `final_claim == f_open·W(r)`.

Collapses the `s` opens → 1 sumcheck + 1 open. It is **purely additive**:
the per-chain `comm_a/b_shifted` are untouched, so `curr_batch` and the
range checks are unaffected (no cascade — that concern was a *different*,
compact-fresh-commit design). Soundness surface is exactly that of
`aprev_batch` (the `W`/point mapping), not the core commitment/identity
structure.

**The real (performance, not soundness) caveat:** `combine_commitments`
concatenates *row*-commitment vectors, so the stacked poly `F` is laid as
full Hyrax rows (`num_cols = 2048` wide). When `2^(n−tk) < 2048` (every
bench config: final poly size `≤ 32`), each chain block is zero-padded to
2048, so the batch's sumcheck + open run over `s_pad·2048` mostly-zero
entries instead of `s` tiny single-row opens. Whether that's a net win at
bench sizes is **empirical** (measure the `chain_openings` span
before/after). The free-concat stack is a *clear* win only at
`n − tk ≥ 11` (≈ 2¹⁶+ constraints), where final polys span full rows and
the per-open cost is real. Padding the chain count to `s_pad = next_pow2(s)`
uses copies of chain 0 (weight 0 in `W`, excluded from the claim) so the
verifier reconstructs the stacked commitment with no identity elements.

**Risk:** low — localized to the final-open path, mirrors the existing,
tested `aprev_batch`; no new commitment, no consistency proof, no cascade.

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
