# IntMod-Spartan deferred work

Running list of optimizations, hygiene items, and tests we've identified but
not yet implemented. Each entry links to its phase of origin and an estimated
size; promote into a phase plan when picked up.

## Zinc+ comparison & performance recovery (2026-06-24)

Full head-to-head in [zincplus_comparison.md](zincplus_comparison.md) (measured
single-thread numbers, mechanism, caveats). Key takeaways for follow-up:

- **~2× prover regression vs `e9e9be5` — NOT recoverable from a branch
  (checked 2026-06-24).** `main` prove-proper is ~3.6 s on multiswap-2¹³; the
  collaborator's out-of-repo `e9e9be5` (paper figures) was ~2× faster. The 2×
  was a real prior measurement (gen/commit ruled out), but the faster code is
  **not in the repo**: `e9e9be5` is a dangling local-only commit (not an object
  here, not on origin). Checked all candidate branches — `msm_opt` is a year-old
  MSM tweak (behind 158); `improv2`'s perf commits (`z_buffer`/`matvec` caches,
  `Parallelize NIFS`, digest streaming) are **already in main**. So there is no
  unmerged "faster" branch. Recovery requires the collaborator to push
  `e9e9be5` (`git push origin e9e9be5:refs/heads/recover` from their local repo)
  or recall the change — otherwise drop the chase and optimize `main` directly
  (commit MSMs ~50%, the GKR).
- **Commits (Pedersen MSMs) are the Brakedown lever — NOT the range check.**
  The range check is ~48% of prove, but its cost splits into a *commit* part
  (`rc_chunk_commit` Pedersen MSM, 1.33 s) and the *GKR* part (`rc_logup_gkr`,
  655 ms). Mod-PCS-over-Brakedown replaces the Pedersen-MSM commits
  (`wq_commit` ~855 ms + `rc_chunk_commit` ~1.33 s) with hash-based commits →
  projects prove toward ~2.5–2.9 s (~1.5×). **The range check itself stays** —
  it is load-bearing for soundness (proves committed integers are bounded;
  without it the mod-random-prime reduction is unsound). Brakedown only speeds
  its *commitment* step, not the GKR/bound logic. Same rewrite as
  `brakedown_design.md`.
- **Don't chase Zinc+ on ECDSA.** They specialize to native q=p (no range
  checks) → 23×. Point "we win" at native-emulation (arkworks/xJsnark, the
  ~10M→8k collapse); frame Zinc+ as concurrent complementary design (their fast
  prover vs our ~7× verify + ~1000× proof-size wins).

## Multi-threading: the GKR is the scaling bottleneck (2026-06-24, deferred)

Measured MT prover speedups (this machine, ~14 cores): **multiswap 2¹³/2048-bit
scales 3.2×** (4.35 s → 1.36 s) but **ECDSA 2¹³/256-bit only ~1.2×** (548 ms →
452 ms). The difference is *what dominates*: MT works great when the prover is
**MSM-bound** (MSM commits parallelize 7–10×: `rc_chunk_commit` 1330→137 ms,
`wq_commit` 855→~125 ms) and stalls when it's **GKR-bound**. Small-field circuits
(ECDSA, 256-bit) have small MSM work, so the GKR + serial overheads dominate.

The laggard is **`rc_logup_gkr` (the LogUp-GKR range check): ~1.45× MT** (655→450
ms), capping prove-proper at 2.9×. The round-polynomial *is* already parallel
(`logup_gkr.rs` `into_par_iter` ~line 337, above `PAR_THRESHOLD`), so it is **not**
a missing `par_iter` (a `par_iter` on the table-build at 797/805 was a wash —
reverted). The limit is **structural**: the range check proves *many* independent
fraction-trees (`chunk_vals_all` + `top_vals_all`, one per chunk batch), and the
loop at `logup_gkr.rs:795` runs them **strictly serially** because each
`gkr_prove` threads the shared Fiat–Shamir transcript — N sequential GKRs, each
only ~1.45× internally.

**Deferred fix — batch the per-witness GKRs into one combined sumcheck.** Build
all witness trees' levels in parallel (`par_iter` over witnesses — pure
computation, no transcript), absorb all roots, then run **one** combined
layer-sumcheck over all witnesses via a random-linear-combination of their
per-layer claims (standard batched-GKR). Gives one set of sequential layers with
**N× parallel width** and removes the serial witness loop → lifts ECDSA toward
~3× and pushes multiswap past 3.2×. Soundness-sensitive: the batched sumcheck +
FS reorder + re-verifying range-check soundness and the `logup_gkr` tests.
**Focusing on single-thread for now** — this is the multi-thread lever when we
return to it.

## Phase 1 follow-ups (identified during 2026-05-26 audit)

### Performance

- **`eval_public_at` is O(2^num_vars).** `src/imod_spartan.rs:373` builds the
  full eq table and dot-products against a length-`num_vars` public vector. Plain
  Spartan uses `SparsePolynomial::new(num_rounds_y - 1, X).evaluate(&r_y[1..])`
  which is O(|X|). Invisible at `num_io = 0` (current benchmarks); becomes a
  verify-time bottleneck once we have circuits with non-trivial public input.
  **Fix:** swap to `SparsePolynomial`. Small (~10 lines).

- **No BDDT round-0 optimization in inner SC.** Plain Spartan's
  `spartan.rs:323-404` computes the inner SC's round-0 polynomial manually
  using delayed-reduction accumulators, saving one full `prove_quad` round.
  imod_spartan just calls `prove_quad` over all `num_rounds_y` rounds.
  Expected savings: ~5-10% on prove time once we hit sizes where SC cost
  dominates. **Fix:** lift the BDDT logic into imod_spartan's inner SC
  prepare-phase. Medium (~80 lines).

- **No batched PCS open.** Two `E::PCS::prove`/`verify` calls (w at `r_y[1..]`,
  q at `r_x`) instead of a single batched open at multiple points. If the
  underlying PCS supports multi-point batching, savings are constant-factor on
  prove and verify; proof size shrinks by ~30 KB. **Fix:** depends on the PCS
  trait — would need a batched-open method exposed. Larger change.

- **`bind_abc` allocates a fresh Vec each call.** `src/imod_spartan.rs:348`.
  Trivial to take `&mut Vec` and have caller pre-allocate once. Useful when
  we add a `PrepSNARK`-style buffer-reuse layer. Small (~10 lines).

- **`evaluate_matrices` uses plain O(nnz) sum.** `src/imod_spartan.rs:397`.
  Same complexity as `SplitR1CSShape::evaluate_with_tables_fast`, just less
  optimized loop structure. Lift the upstream helper into a flat-shape
  variant once it matters. Medium.

- **No `SpartanPrepSNARK`-style buffer reuse.** Plain Spartan threads a
  `prep_snark` struct containing pre-allocated scratch (`scratch_az/bz/cz`,
  `z_buffer`, `evals_rx_buffer`) across prove calls. imod has no equivalent.
  At current bench sizes the allocations don't dominate; revisit if a
  flamegraph shows allocation overhead. Larger refactor (API change).

- **No incremental SpMV with cached partial products.** Plain Spartan's
  `multiply_vec_incremental_into` caches partial matvec results for the
  precommitted witness segment. imod has no precommitted segment so this
  doesn't apply directly; if we add commit-and-prove later it would.

### Tests

- **No test with `num_io > 0`.** All three positive tests use `num_io = 0`,
  so `eval_public_at` is never exercised with actual public input and the
  `eval_w = (eval_z - r_y[0] * eval_x) / (1 - r_y[0])` recovery is degenerate.
  **Fix:** add one positive test with non-trivial X. Small (~30 lines).

- **No test for `eval_w` / PCS-opening tampering.** The existing tamper test
  only flips `v_q`. Should also mutate `eval_w`, `eval_arg_w`, `eval_arg_q`
  to confirm those rejection paths are live. Small.

- **All tests use tiny dimensions.** num_vars ∈ {4, 8}, num_cons = 2. Inner
  SC only runs 3-4 rounds. Bugs that manifest only at larger sizes wouldn't
  be caught. Add at least one test with num_cons ≥ 2^8 or so. Small.

### Code hygiene

- **`absorb_shape` is now obsolete and removed.** Replaced with `vk_digest`
  in commit `edfd94d`. No follow-up needed; noted for completeness.

- **No `validate` method on `IntModR1CSInstance`.** Plain Spartan has
  `R1CSInstance::validate(S, transcript)` (`r1cs/mod.rs:1490`) that checks
  commitment sizes and absorbs into the transcript. imod inlines this at
  the top of prove/verify. Refactoring into a `validate` method would
  mirror plain Spartan's structure more closely and centralize the
  PCS::check_commitment calls. Small.

- **Defense-in-depth debug_asserts.** Add `debug_assert!(U.x.len() ==
  shape.num_io)` and similar at the top of prove/verify. Catches misuse
  earlier than the eventual mismatch. Trivial.

- **`bind_abc` should `debug_assert!(num_cols <= 2*num_vars)`.** Currently
  relies on the constructor invariant (`num_vars >= 1 + num_io`); if a
  future change relaxes that invariant, `bind_abc`'s `.resize(2*num_vars,
  ZERO)` silently truncates. Trivial.

- **Domain separator `b"IntModSpartanSNARK"` could be versioned.**
  Something like `b"IntModSpartanSNARK-v1"` so future protocol revisions
  don't risk transcript collision. Trivial.

## Phase 2 follow-ups (retroactively backfilled 2026-06-01)

### Performance

- **`IntModR1CSShapeModp` matrices are raw COO `Vec<(usize, usize, BigUint)>`.**
  We couldn't reuse `r1cs::SparseMatrix<F: PrimeField>` because `BigUint`
  isn't `PrimeField`. The COO storage means `multiply_vec`, `bind_abc`, and
  `evaluate_matrices` walk every entry linearly per call instead of using
  CSR row offsets. Cost grows linearly in `nnz(A) + nnz(B) + nnz(C)`. For
  the toy tests (a handful of entries) it doesn't matter; for any
  realistic SNARK this is a measurable regression vs Phase 1.
  **Fix:** parameterize `SparseMatrix` over a general scalar type, or
  introduce a `SparseMatrixModp<M>` variant with CSR over `BigUint`. Medium.

- **Matrix reduction `BigUint → DynPrime` isn't cached.** Every `prove` /
  `verify` reduces shape matrices/mods from `BigUint` → `DynPrime` mod the
  newly-sampled `p`. For repeated proving over the same shape (the typical
  case at the application layer), this is wasted work. The reduction depends
  on `p` which is per-session, so it can't be precomputed at setup. But it
  could be cached *per `prove` call* across the multiple uses (`mods_p`,
  `a_p`, `b_p`, `c_p`) — which we mostly already do, but the helper
  `biguint_to_scalar::<M>(v, params)` is called per element on three
  matrices and three vectors with no batch parallelism. **Fix:** rayon-
  parallelize the reduction loops.

- **`eval_public_at` builds the full eq-table even for `num_io = 0`.**
  Inherited from Phase 1 — same `SparsePolynomial::evaluate` swap applies.
  See Phase-1 follow-up of the same name.

- **Sumcheck over `DynPrime<4>` is ~3-5× slower than over `t256::Scalar`.**
  `crypto_bigint::FixedMontyForm` doesn't have the ASM optimizations
  halo2curves uses for the static-modulus path. Each `*` `+` `-` goes
  through generic Montgomery reduction. This is the dominant Phase-2 perf
  hit. **Fix:** wait for crypto-bigint to add ASM (issue open upstream), or
  hand-write the Montgomery routines for the common widths.

- **No bench for Phase 2.** `benches/imod_spartan.rs` still targets the
  Phase-1 driver. Per the perf-target memory, we should re-benchmark imod
  vs plain Spartan on shape-matched configs whenever the protocol changes —
  Phase 2 changed it. **Fix:** clone `imod_spartan` bench → `imod_spartan_modp`
  bench. Run it. Add results to a perf log.

### Correctness boundaries

- **Verifier cross-vk panic.** Documented in the `imod_modp_digest_binds_matrices`
  test: verifying a Phase-1-style proof under a wrong vk panics inside
  `crypto-bigint::FixedMontyForm` because the proof's `DynPrime` values
  carry `params_p1` and the verifier reduces shape data with `params_p2`.
  Functionally rejection but ungraceful. **Fix:** wrap the param check
  earlier and return `SpartanError::InvalidSumcheckProof` instead of
  panicking. Small.

- **`DynPrime<4>::from_bytes_reduce` truncation — FIXED (`063aab6`,
  2026-06-11).** History: inputs wider than 256 bits were silently
  truncated, so the Phase-2 Z_p layer bound the *truncated*
  witness/moduli/quotients; honest instances with random >256-bit
  values failed to verify, and the pre-fix MultiSwap bench only
  roundtripped because its synthetic operands satisfied each row as a
  polynomial identity in `m` (preserved by truncation). Fixed by
  MSB-first Horner over 32-byte chunks with `2^256 mod p` weights.
  Verified by the un-ignored regression test
  `tests/wide_value_probe.rs::wide_modulus_roundtrip_random_operands`
  and the wired RSA-2048 roundtrip. Side benefit: 64-byte transcript
  squeezes now reduce in full, satisfying the trait's challenge-bias
  note. Remaining follow-up: the focused soundness review of the
  pre-fix-era reasoning is moot, but the *exact-row convention* it led
  to is now load-bearing — see next item.

- **Exact-row (`m = 0`) convention + bit-gadget soundness note
  (2026-06-11).** The wired exponentiation gadget's binary constraints
  originally used `b·b = b (mod N)`, which admits non-binary solutions:
  benign lifts of 0/1 (e.g. `b = N+1, q = N+1`) and — harmfully —
  nontrivial idempotents of `Z_N`, though exhibiting those factors `N`,
  so the mod-N variant is *computationally* sound for RSA moduli.
  Switched to **modulus-0 exact rows** (`b² = b` over ℤ):
  unconditional, same cost, and safe to reuse for moduli with known
  factorization (the bench's `ℓ` is one — mod-ℓ exponent bits in a
  future Hp gadget MUST use exact rows or bound-2 range segments).
  Convention documented on `IntModR1CSShapeModp`; roundtrip + negative
  tests added (`imod_modp_exact_row_mod_zero_roundtrip`,
  `imod_modp_exact_bit_row_rejects_lift`). Note `m = 1` is the
  degenerate modulus (vacuous row); consider a validation guard.

- **`prove_with_iter` doesn't bind `p_i` into the iteration commits.** See
  the same item in Phase 3 follow-ups — applies to step C only, but it's
  rooted in Phase 2's transcript design (the prime indices aren't labelled).

### Code hygiene

- **Phase-1 `imod_spartan` is no longer the canonical path** but it's still
  in tree (with `_modp` parallel module). Once Phase 3 stabilizes, decide:
  delete Phase 1, or keep it as a reference for `p = q` mode. See the
  `project_phase2_parallel_sumcheck` memory — option B → A migration was
  planned with differential testing; we haven't done either.

- **`bootstrap_params` placeholder.** The Phase-2 driver constructs the
  transcript with `M::bootstrap_params()` (modulus = 3 for `T256DynPrimeEngine`),
  a placeholder never used arithmetically. Wart from `Keccak256Transcript`
  needing *some* `Params` at construction. **Fix:** split `Keccak256Transcript`
  into a byte-only mode + a typed-squeeze mode; bootstrap only needs the
  byte-only mode. Medium.

- **`set_params` is a `Keccak256Transcript`-inherent method, not on the
  trait.** The Phase-2 driver consequently uses `where M: ModEngine<TE =
  Keccak256Transcript<M>>`, tying it to the specific transcript impl. **Fix:**
  promote `set_params` to a method on the trait (with a default no-op for
  static-field cases). Small.

- **`bootstrap_params` + `sample_params` are explicit-impl-required** on
  every `ModEngine`, no default. Static-field engines must spell out `()`
  trivially. **Fix:** restore a default impl gated on `Params: Default`
  (currently dropped to avoid bound-rippling). Small.

- **Proof / shape not `Serialize`.** `IntModSpartanModpSNARK` and
  `IntModR1CSShapeModp` aren't `Serialize`/`Deserialize`. Was OK during
  prototype but blocks any real "save proof to disk" usage. `BigUint`
  already impls `Serialize`; the blockers are the `M::Scalar` and
  `<M::Scalar as SumcheckField>::Params` types in the proof struct.
  **Fix:** add `Serialize` impls. Medium (need to think about whether
  `Params` is part of the proof or recoverable from public data).

- **`SumcheckField::Params` requires `'static`** which is a leak of
  implementation detail — `FixedMontyParams` happens to be `'static` but
  the bound shouldn't be there structurally. Probably can drop. Trivial.

### Tests

- **Two-row + public-IO tests use very small dimensions.** num_vars=8
  exercises the n ≤ k IntEval path only. ~~Step C is not reached~~ —
  resolved by `imod_modp_snark_with_inteval_iteration` (num_vars=256,
  triggers t=1 on the W open). Still no SNARK test with t > 1 (would
  need num_vars >= 2^14 with default k=7); add when relevant.

- **No differential test between Phase 1 and Phase 2.** When p=q is forced
  on Phase 2, the two should give equivalent results on the same toy
  circuits. We haven't checked. **Fix:** write a differential test. Small.

## Phase 3 follow-ups (identified during step A–C implementation, 2026-06-01)

### Performance / structure

- **Phase-2 prover is ~17× plain Spartan at scale (12× verify).**
  Bench sweep at `n_cons ∈ {2^6, 2^8, 2^10}` with `num_vars = 4·n_cons`
  rounded up to next power of two (Apple Silicon, default features,
  2026-06-01):

  | n_cons | Setup (P0/P2) | Prove (P0) | Prove (P2) | P2/P0 | Verify (P0) | Verify (P2) | P2/P0 |
  |--------|---------------|------------|------------|-------|-------------|-------------|-------|
  | 2^6    | 20 / 18 ms    | 7.1ms      | 75ms       | 10.6× | 5.5ms       | 76ms        | 13.9× |
  | 2^8    | 19 / 19 ms    | 7.8ms      | 138ms      | 17.7× | 5.6ms       | 136ms       | 24.4× |
  | 2^10   | 19 / 19 ms    | 12.8ms     | 223ms      | 17.4× | 6.2ms       | 168ms       | 27.0× |

  Observations:
  - Setup is comparable across all three sizes (Hyrax-T256 setup
    dominates, ~19ms).
  - Phase-2 prove ratio grew then stabilized (10.6× → 17.7× → 17.4×).
    The "constant factor" target is roughly 17× at the larger sizes.
  - Verify ratio still climbing (13.9× → 24× → 27×). Phase-2 verify
    redoes all `s × t` chain Hyrax verifies; plain-Spartan verify is
    just two Hyrax::verify calls so it stays nearly flat in this range.
  - Plain Spartan is sub-linear here (7.1→7.8→12.8); per-constraint
    cost crosses into linear-scaling territory around `n_cons = 2^10`.
  - Phase 2 scales roughly linearly (prove grew 1.84× then 1.61× for
    each 4× witness increase). Ratio likely peaks ~20-25× and stays
    there.

  At `(2^10, 2^12)`, Phase-1 imod numbers aren't measured but should
  match plain Spartan within ~5-10% (the P1 vs plain delta at `2^6`
  was only ~20% on verify).

  **Refresh after Phase-3 step D5 + stacking refactor (2026-06-02):**
  D5 first shipped with `1 + 2·s·t` separate range-check arguments
  (one per polynomial), which landed prove/verify at ~40-100× plain
  Spartan. The follow-on commit (`99fc22f`) restructured into
  `1 + 2t` homogeneous batches — `f_limb`, and per iteration `j` an
  `a_j` batch (all `s` chains) and a `b_j` batch. Combined with
  Milestone 1 (`587569c`, parallel hot loops + `t256_q` memoization
  + hoisted divmod) the prover and verifier each dropped ~50%:

  | n_cons | Setup | Prove (separate → batched) | Verify (separate → batched) | P3/P0 prove | P3/P0 verify |
  |--------|-------|-----------------------------|------------------------------|-------------|---------------|
  | 2^6    | 18 ms | 285 → 149 ms (-48%)         | 254 → 134 ms (-47%)          | 21×         | 24×           |
  | 2^8    | 18 ms | 632 → 286 ms (-55%)         | 535 → 242 ms (-55%)          | 37×         | 43×           |
  | 2^10   | 18 ms | 863 → 429 ms (-50%)         | 621 → 299 ms (-52%)          | 34×         | 48×           |

  Still well outside the Phase-2 ratio (~17×/~25×), but back inside
  "ambitious constant factor" territory. The b_j batch is still the
  worst per-group cost — bound `2q/P` ≈ `2^227`, ~256 padded bits per
  coefficient — but it now amortizes over all `s` chains via one
  shared bit commitment and one shared sumcheck per iteration.

  Next perf levers worth profiling:
  - **Share the witness commitment with the `f_limb` range-check chunk
    commitment (`log_t = CHUNK_BITS = 16`).** When the limb width equals the
    range-check chunk width (16), each limb *is* one chunk, so
    `limb_split(w)` equals the chunk decomposition — i.e. `comm_w` (and
    `comm_q`) and the range check's `f_limb` chunk commitment commit the
    *identical* polynomial. Use `log_t=16` and reuse `comm_w`/`comm_q` as the
    `f_limb` chunk commitment instead of committing it again. Total
    range-check chunks are `log_t_f/16` regardless of `log_t`, so the GKR
    cost is unchanged; this purely drops the duplicate MSM. Per-value commit
    entries (msshape, `log_t_f=256`): `log_t=32` no-share = 8 limbs + 16
    chunks = 24; `log_t=16` + share = 16 (one commit). The witness commit is
    ~21% of single-threaded prove, so collapsing its duplicate is real.
    Caveats: (1) the witness-commit layout must be made to match exactly what
    the range check consumes (same `ck`, ordering, padding — at `log_t=16`
    `stride=1`, so they can coincide); (2) only `w`/`q` share — the `a_j`/`b_j`
    IntEval chains keep their own range-check commits; (3) safe only post the
    `is_small = log_t <= 64` fix (commit `f045217`). Needs a verify-inclusive
    A/B (bigger single shared commit vs dropping a separate one). Idea raised
    by the user 2026-06-18.
  - **Batched Hyrax opens inside a `BatchRangeCheck`.** Each batch
    still does `N` separate Hyrax::prove calls for the value-poly
    openings at the shared `r_v_within`. Multi-point batched open
    would collapse them.
  - **Cross-batch bit-comm sharing.** `a_j` and `b_j` at the same `j`
    have identical `n_values` but different `log_bound` — could share
    a single bit commitment with per-segment weight masking, similar
    to how the in-batch poly stacking works today.
  - **Bigger jump:** stack the `1 + 2t` batches into a single
    batched argument across all (bound, size) groups using per-
    segment weights — closer to the paper's `rbatchrange^{s·t}^2`.
    Needs a uniform stride = `max(log_T, log_p+1, LOG_Q-log_p+1)`
    and group-indexed weight masking.

  Likely Phase-2 cost attribution (rough estimates, not yet profiled):
  - **Step-C Hyrax openings dominate.** Per Mod-PCS open with n > k:
    `s` final-remainder opens + `s × t × 3` identity-check opens.
    For `(2^10, 2^12)` with `s=9, t=1`: 36 Hyrax opens per Mod-PCS
    prove × 2 Mod-PCS proves per SNARK = ~72 opens at ~3-5ms each →
    ~200-360ms. Matches observed 223ms prove time.
  - **DynPrime arithmetic in the sumcheck** (~3-5× per op vs the
    static `t256::Scalar`). Probably the second-largest contributor.
  - **Miller-Rabin prime sampling: small.** ~88 × 30µs for the big
    `p`, ~18 × 21 × 1µs for the small primes, total ~3ms (1-4% of
    prove). Not a perf priority.

  Caveat on the comparison: P1/P2 imod synthetic shape has 3 nnz/row
  (one per matrix); plain Spartan's bellpepper-synthesized multiplication
  uses denser matrices. Proof-size comparison is therefore not
  apples-to-apples; constraint count is the meaningful axis until we
  shape-match more carefully.

- **Cross-cutting opportunity: batching at every layer.** Right now we
  do nothing in batches — every commitment, every sumcheck, every
  Hyrax opening goes through its own protocol invocation. This is a
  large fraction of the Phase-2 overhead. Batching opportunities, in
  order of (estimated) impact:

  - **Batched Hyrax openings (biggest win).** Per Phase-2 SNARK we do
    ~72 separate `Hyrax::prove` / `Hyrax::verify` round-trips (see the
    cost attribution above). Many of them are at the *same point* on
    *different commitments* (e.g. in step C, `a_{j-1}`, `a_j`, `b_j`
    are all opened at γ-related points), or at *different points* on
    the *same commitment* (the `s` final-remainder opens of `a_t`,
    one per prime). Both forms admit standard batching via random
    linear combinations of the eval claims, reducing `O(s × t)` opens
    to `O(s + t)` or even `O(1)` per oracle. Per-prime work is
    embarrassingly parallel too. Combined: probably ~5× prove speedup.

  - **Batched polynomial commitments.** The SNARK does separate Hyrax
    commits to `w` and `q`; IntEval step C does separate commits to
    `a_j` and `b_j` per iteration. A multi-row Hyrax commit (laying
    polynomials side by side and committing once) saves redundant
    setup work and amortizes MSM. Modest perf win, bigger proof-size
    win.

  - **Batched sumchecks.** The outer cubic SC and inner quadratic SC
    run sequentially. Standard SNARK practice: run multiple
    independent sumcheck instances in parallel via RLC of the running
    claims. Less straightforward when the integrands have different
    degree (cubic vs quadratic) — usually you reduce both to a single
    common-degree sumcheck via a degree-lifting step. Plain Spartan
    already does some of this; we don't. Worth a small writeup of
    what's available before implementing.

  - **Batched range checks (Phase-3 step D — partially done as of
    `99fc22f`).** Range checks on every `a_j` and `b_j` are now
    bundled per-iteration into homogeneous batches (one batch per
    `s` chains' `a_j`, one per `b_j`), plus a single batch for
    `f_limb`. The paper's full `rbatchrange^{s·t}^2` would fuse all
    `1 + 2t` batches into one global argument; see the D5 section
    below for the remaining stacking opportunities.

  Most of these change the *protocol structure* rather than just hot-
  path inlining, so they need to land carefully against the IntEval
  paper's soundness analysis. Worth doing once Phase 3 step D is in
  place; doing them piecemeal before D risks rework.

- **`IntEvalArgument` is huge under default params.** Step C produces, per
  prove call: `s` chains × `t` iterations × 2 polynomial Hyrax commits, plus
  `s × (3t + 1)` Hyrax openings (each carries `f_y`, `blind_eval`, and a
  Hyrax eval-argument). The batching items above also shrink the proof
  size. Plus a minor item: compact the `BigInt int_v_prime` serialization
  (currently sign byte + 8-byte LE length + LE magnitude; could be
  tighter).

- **Per-`p_i` matrix-style work is not parallelized.** The `s` chains are
  embarrassingly parallel — each is independent until γ is sampled. **Fix:**
  rayon-parallelize the per-chain phase 1 loop in `IntegerModPCS::prove`.
  Same for verify. Probably ~3-5× speedup on prove for default `s = 10`.

- **`mle_evaluate_fq` walks the full eq-table every call.** Each γ-prefix
  opening recomputes the eq-table from scratch. For step C, each chain
  recomputes the same γ-prefix table many times. **Fix:** cache the
  γ-prefix eq-tables once per (prefix_len), reuse across chains.

- **Integer partial-eval allocates a full chi table.** `integer_partial_evaluate_top_k`
  builds a `2^k`-length `Vec<BigInt>` of chi values, then dot-products. For
  larger `k` this is memory-heavy. **Fix:** streaming computation — accumulate
  the output directly without materializing chi[]. Minor; only matters at
  large k.

- **Small-prime rejection sampling does up to 16 bytes of work per try, but
  squeezes 64.** `sample_small_prime` squeezes a full 64-byte transcript
  block per candidate. The remaining 48 bytes are wasted. **Fix:** absorb-
  once, candidate-from-bytes-with-counter, retry without re-squeezing.
  Tiny perf win; probably not worth the complexity.

- **`shift_b` per-call recomputes `t256_q()`.** Cheap but unnecessary —
  `q` is a constant. **Fix:** `once_cell::sync::Lazy<BigUint>` for `q`.

### Correctness boundaries / TODOs

- **`comm_eval` / `blind_eval` are dead weight on the Mod-PCS trait
  surface for `IntegerModPCS`.** The `prove`/`verify` bodies ignore
  them (`_comm_eval`, `_blind_eval`). The SNARK driver creates them
  anyway via `M::ModPCS::commit(&ck_s, &[eval_value], …)`, which has
  two awkward consequences:

  - The "eval value" can be any F element (a Z_p eval ~128 bits) but
    `Mod-PCS::commit` is supposed to commit *integer-bounded* values
    (`< T_f`). Step D2 added a `v.len() == 1` stopgap that skips
    limb-splitting for single-value commits so the assertion doesn't
    trip; the stopgap muddies the semantic of `commit` ("polynomial of
    bounded integers" or "single field element").
  - We pay a Hyrax commit (one MSM) per opening for a value that
    isn't actually verified through the trait's `comm_eval` channel.

  **Fix:** drop `comm_eval` and `blind_eval` from the `prove`/`verify`
  trait surface (or move them behind a non-default associated type so
  hash-based / non-Pedersen Mod-PCS impls don't have to fake them).
  Ripples through `TrivialModPCS` (which does use them via the
  underlying `E::PCS::prove`) and the SNARK driver. Once dropped, the
  `v.len() == 1` stopgap in `commit` goes away. Medium.

- **Limb-splitting + range check on limbs not implemented (Phase-3 step
  D).** Without limb-splitting we're stuck at `T = T_f` (the bound on
  polynomial coefficients equals the bound on each "limb"), which keeps
  the Partial Eval Norm Bound `2^k · P^k · max(T, P) ≤ (q-P)/2`
  reachable only for small `T_f`. To soundly support large-norm
  polynomials we need both pieces from the paper §4.1–4.2 together:

  1. **Limb-split commit.** Split integer `f` with bound `T_f` into a
     polynomial `f_limb` of size `2^(num_vars + numlimb_var)` where
     each slot holds a limb in `[0, T)` and `numlimb = ⌈log_T(T_f)⌉`.
     Commit `f_limb` via the underlying F PCS.
  2. **Reduction sumcheck at eval.** The Mod-PCS eval protocol gains
     a sumcheck that reduces an eval claim about `f` to an eval claim
     about `f_limb`: `sum_{k ∈ {0,1}^numlimb_var} limb(k) · f_limb
     (int_r, k) ≡_p int_y`, where `limb` is the public weight vector
     `[1, T, T^2, …, T^{numlimb-1}]`.
  3. **Batch range check on limbs.** Each committed limb must satisfy
     `|limb| < T`; the paper batches all `s × t` limbs across all chains
     into one `rbatchrange{s·t}^2` argument. Without this the limb-
     split protocol is still unsound (prover can put arbitrary values
     in limbs).

  Step D split into 5 sub-tasks: D1 (numlimb derivation + LimbSplit
  helper), D2 (commit refactor, numlimb=1 no-op), D3 (reduction
  sumcheck, numlimb=1 no-op), D4 (enable numlimb > 1 + tests), D5
  (batch range check argument). D5 is the algorithmically heaviest;
  worth doing the range-check batching together with the broader
  batching cross-cutting refactor (see Performance / structure above).

- **D5 followups (recorded 2026-06-02 after D5.2 lands; updated as
  follow-on work shipped).**

  - **(Partially done) Stack range checks into combined arguments.**
    D5.2/.3/.4 initially shipped `1 + 2·s·t` separate range-check
    arguments (one per polynomial). `99fc22f` restructured to
    `1 + 2t` *homogeneous batches*: `f_limb`, then for each iteration
    `j` an `a_j` batch (all `s` chains) and a `b_j` batch. Each
    batch is one `BatchRangeCheck` — one bit commitment, one
    bit-validity zerocheck, one value-reconstruction sumcheck, and
    `N + 2` Hyrax openings. ~50% perf win (see bench refresh above).

    Remaining stacking opportunities, in increasing scope:
    1. **Across (a, b) at the same `j`.** Same `n_values`, different
       `log_bound` — could share a single bit commitment with
       per-segment weight masking.
    2. **Across all iterations + f_limb.** Stack the `1 + 2t`
       batches into one global batched argument with per-group
       weight masking. Uniform stride =
       `max(log_T, log_p+1, LOG_Q-log_p+1)`. This is the paper's
       full `rbatchrange^{s·t}^2`.
    3. **Batched Hyrax opens** inside each `BatchRangeCheck` — the
       `N` value-poly openings at `r_v_within` are at the same point;
       a multi-point batched Hyrax::prove would collapse them.

  - **Range check is semantically a commitment obligation but lives
    eval-side.** The `f_limb < T` check could be in `commit()`
    (binding `Commitment` to its bound), but that would require
    seeding the range-check transcript from the commitment bytes,
    locking the proof into the `Commitment` struct, and blocking the
    future stacking-with-a_j/b_j optimization above. We chose to
    keep it in eval-side under `IntEvalArgument` so all three range-
    check categories share one transcript and can be stacked later.
    If the paper's commit-binding gets stricter, or if we ever serve
    commitments without an eval proof attached, revisit.

  - **Range-check sub-transcript runs in `t256::Scalar` (the Hyrax
    base field).** The bit polynomial is Hyrax-committed, so its
    openings return `t256::Scalar` values; the sumcheck final claim
    must be in the same field so the reconstruction `eq(r,τ) · (bit(r)² -
    bit(r))` matches what the Hyrax opening produces. Concretely the
    prover spawns a `Keccak256Transcript<T256HyraxEngine>` seeded
    with bytes from the parent (DynPrime) transcript, runs the whole
    range-check protocol there, and the parent is unaffected after
    the seed squeeze. If the parent is changed downstream we must
    absorb the BatchRangeCheck back into it.

- **`IntegerModPCS` types leak the underlying PCS (Hyrax + T256).**
  `SmallPrimeOpening`, `IterationOracles`, `BatchRangeCheck`, and the
  helpers `hyrax_open_at` / `hyrax_verify_open` are all hardcoded to
  `<Hyrax as PCSEngineTrait<T256HyraxEngine>>::{Commitment, Blind,
  EvaluationArgument}` and `t256::Scalar`. The `ModPCSEngineTrait`
  surface itself stays PCS-agnostic (associated types are opaque
  `Self::Commitment` etc.), but inside `IntegerModPCS` the
  implementation is Hyrax-only — a future KZH or KZG-backed integer
  Mod-PCS would need to duplicate most of `integer_modpcs.rs`.
  D5.2–D5.4 marked their new struct fields `pub(crate)` so the
  module boundary stays opaque; full genericization is its own
  refactor (a sweep to `BatchRangeCheck<E, P>`, generic
  `prove_batch_range_check`, etc.). Worth doing once a second
  Mod-PCS backend is actually on the roadmap.

- **`prove_with_iter` chain commitments aren't bound to `p_i`.** The
  transcript absorbs `comm_a_shifted` / `comm_b_shifted` but not the prime
  they were computed against. Two chains with the same commits but different
  primes would be indistinguishable to the FS chain. **Fix candidate:**
  absorb `p_i` (or a label `i` + the actual p_i bytes) before the iteration
  commits. Probably already implicit via the prime-sample ordering, but
  worth double-checking.

- **Verifier cross-vk panic still unfixed.** From Phase 2's
  `imod_modp_digest_binds_matrices` test note: verifying a proof under the
  wrong vk currently panics inside crypto-bigint when `DynPrime` ops hit
  mismatched `FixedMontyParams`. Should convert to a clean `SpartanError`.

- **`scalar_to_balanced_int` only used for step B's final eval.** Step C's
  identity check and final remainder use `shift_a`/`shift_b` subtraction
  in F instead — no balancing needed (the F-arithmetic stays positive
  thanks to the shifts). Code path could be cleaned up.

### Param derivation

- **`IntEvalParams` defaults are baked into `IntegerModPCS::setup` and
  the SNARK driver has no way to override them.** The trait-level setup
  uses `(DEFAULT_LOG_T_F = 32, DEFAULT_K = 7)` for any call. Three
  concrete problems with this:

  1. **`T_f = 2^32` is a silent correctness boundary.** `commit` accepts
     any `BigUint` via `from_uniform`; nothing enforces `|v| < 2^32`.
     If an application uses wider witness values, the soundness analysis
     (which assumes `|T_f| < 2^32` for the Partial Eval Norm Bound) breaks
     silently. This intersects with the deferred range-check work but is
     also a documentation / API-contract issue.

  2. **`k = 7` is a tuning knob, not a security parameter.** The paper's
     table 4.4 shows multiple valid `(k, log_P, s)` configs per shape with
     different "extra commit cost ratios" — picking `k` is a trade-off the
     application should be able to make. We currently pick `k = ⌈log λ⌉`
     and that's that.

  3. **No application path to set params.** `IntegerModPCS::setup_with_params`
     exists for the explicit-override case, but the SNARK driver
     (`IntModSpartanModpSNARK::setup`) doesn't expose it — it calls the
     shape's `commitment_key()` which calls the trait `setup` with the
     hardcoded defaults. An application that wants `T_f = 2^64` has no
     clean path through the existing SNARK API.

  **Fix:** extend `IntModSpartanModpSNARK::setup` (and the shape's
  `commitment_key()`) to accept an optional `IntEvalParams` (or a more
  abstract `ModPCSConfig`-style struct) and thread it down. Application-
  layer `setup_with_params` then composes naturally. Medium.

- **`IntEvalParams::derive` picks the smallest valid k from `k_start = log λ`
  upward.** This isn't always the optimal choice — sometimes a *larger* k
  gives fewer iterations and better commitment cost (per the table in §4.4).
  **Fix:** rank by `compute_params.py`-style `extra_commit_ratio` and pick
  the best.

- **`validate` uses the paper's strict text formula for Soundness 1, which
  the paper's *own* table rows don't satisfy.** We've documented this — the
  script uses a tighter analysis. Our strict check is conservative (rejects
  valid-per-the-script configs). **Fix:** port the script's formula.

### Code hygiene

- **The IntEval impl is concrete to `T256DynPrimeEngine`.** No abstraction
  over which underlying F-PCS to use. Once we have multiple `ModEngine`s,
  consider generalizing — but probably not before then.

- **No serialization tests for `IntEvalArgument`.** The struct includes
  `BigInt` (Sign + magnitude) and `Vec<HyraxCommitment>` etc; nothing
  exercises that the bincode roundtrip works. Add a smoke test.

## Phase 2+ design considerations: keeping ModPCSEngineTrait hash-PCS-friendly

When fleshing out `ModPCSEngineTrait`'s method signatures (step 2 onwards),
avoid baking in Pedersen-specific assumptions. The Mod-PCS may eventually
wrap a FRI / Brakedown / Ligero / BaseFold / hash-tree construction, and the
trait surface must not exclude those.

Things to watch for:

- **Blinds.** Pedersen needs `&Blind` everywhere because the commitment is
  `Commit(value, blind)`. Hash-based PCSes are non-hiding and have no blind
  analog. **Plan:** drop `&Blind` from `ModPCSEngineTrait::commit/prove/verify`
  signatures. If a group-based ModPCS impl needs blinds, it carries them
  internally (or via an extension trait), not through the universal interface.

- **Separate eval commitment key (`ck_s`).** Plain Spartan uses a size-1
  Pedersen commitment for the claimed evaluation (`eval_w`). Hash-based PCSes
  don't need this — they send the scalar plus a hash. **Plan:** no `ck_s` in
  `ModPCSEngineTrait`; if a Pedersen-based ModPCS needs one, internal detail.

- **Group-additive helpers.** `commit_without_blind`, `commit_incremental`,
  `combine_commitments`, `combine_blinds`, `fold_commitments`, and the
  `FoldingEngineTrait` extension all exploit Pedersen homomorphism. Hash-based
  PCSes can't satisfy these. **Plan:** leave them out of `ModPCSEngineTrait`.
  If group-based impls want them, add a separate extension trait
  (`GroupModPCSEngineTrait` or similar) bounded on the homomorphic property.

- **`is_small: bool` hint on commit.** MSM-with-small-scalars optimization,
  Pedersen-specific. **Plan:** drop from `ModPCSEngineTrait` surface.

- **`type CommitmentKey: ()` possibility.** Some hash-based PCSes (Brakedown
  with no trusted setup, BaseFold) don't need a structured setup. **Plan:**
  don't force `CommitmentKey` to be non-trivial; the trait should accept
  `type CommitmentKey = ()`.

- **`EvaluationArgument` size.** Group-based proofs are O(√n) group elements;
  hash-based are O(query · log n · path). Trait-level: no issue. Benchmark/
  test level: don't bake in expected proof-size constants.

- **`type Blind: Default` if we keep blinds anywhere.** For backward compat
  with group-based PCSes, if we end up needing a `Blind` slot somewhere, make
  it `Default` so non-hiding impls can use `()`.

Sanity check before locking in the step-2 method signatures: look at
Plonky2, RISC0, and Binius's PCS trait conventions and verify our shape
doesn't accidentally exclude them.

## IntEvalParams (k, T) retune — measured sweep (2026-07-13)

Full (size × k × log_t) sweep on msshape (vars 2^11–2^14, 256-bit; KSWEEP=1
harness in `benches/imod_spartan_modp.rs`) + MultiSwap 2^13 (2048-bit; KSWEEP=1
in `benches/multiswap_modp.rs`), single-threaded, every config verified:

- **T = 2^64 (log_t=64) is fastest at every size and both bit-widths** —
  halving numlimb beats the larger `s`. T=2^16 and T=2^32 are dominated.
- **Under T=2^64 the optimal k ≈ 9 at every size** (flat basin k=8–10, ±2%).
  No per-size k table needed. The apparent size-drift of the optimum seen
  earlier was a T=2^32 phenomenon (and the old 2^12 datapoint predated the
  LogUp-GKR merge). Optimum is insensitive to bit-width (256 vs 2048 agree) —
  `derive`'s (log_p, s) depends on n only logarithmically.
- **Defaults changed: `DEFAULT_K = 9`, bench `LOG_T/MSSHAPE_LOG_T = 64`,
  ECDSA derive → (256, 64, 9, ·).** New single-thread headlines:
  ECDSA MSM 2^13: prove **386 ms** / verify **24 ms** (was 545/43, −29%/−44%);
  MultiSwap 2^13: prove **3.11 s** / verify **42.5 ms** (was 4.35 s/71 ms,
  −29%/−40%). vs Zinc+ MulModN (1.10 s): prove gap now ~2.8×, verify win ~11×.
- This is very plausibly (a chunk of) e9e9be5's "per-input-length IntEvalParams
  optimization" — recovered by measurement, no protocol change, soundness
  unchanged (`derive` validates every config; k=14+/T=64 correctly rejected).
- **T=2^16 witness-commit reuse: implemented, measured, REVERTED.** Reusing the
  witness commitment as the range-check chunk commitment (valid when
  log_t == CHUNK_BITS = 16) is correct but dominated: T=16 doubles numlimb and
  the extra limb bulk outweighs the saved chunk-commit MSM (ECDSA wash 434 vs
  437 ms; multiswap 3.54 vs 3.44 s; both lose to T=64's 386 ms / 3.11 s). Don't
  redo. (If CHUNK_BITS ever grows to 32/64, revisit — reuse at T=64 would not
  double limbs.)

## Regenerated fig:nativeoverhead data — post-re-tune (2026-07-14)

Same-day, same-toolchain (rustc 1.97), single-threaded criterion pairs
(`imod_spartan_modp -- msshape` vs `spartan_synthetic -- msshape`; both prove
regions include witness gen + commit):

| cons | imod prove (T=2^64, k=9) | native prove | overhead | imod verify | native verify |
|---|---|---|---|---|---|
| 2^10 | 204 ms  | 18.8 ms | **10.8×** | 34.9 ms | 14.5 ms |
| 2^12 | 565 ms  | 45.1 ms | **12.5×** | 39.2 ms | 15.1 ms |
| 2^14 | 1.99 s  | 94.6 ms | **21.1×** | 35.6 ms | 16.4 ms |

- Down from ~23–35× pre-re-tune; paper (`e9e9be5`) claimed ~9× — we are within
  ~1.2–1.4× of that at 2^10/2^12; the 2^14 point remains ~2.3× off.
- **k=9 confirmed optimal at 2^15 vars too** (sweep: k=9 1968 ms < k=8 2051 <
  k=10 2010) — the 2^14 gap is NOT parameter mistuning but genuine
  super-linear scaling of the range-check/opens machinery vs the baseline;
  that residual is the remaining e9e9be5 delta and/or the Brakedown lever.
- Verify overhead is now only **~2.2–2.4×** (imod verify ~flat 35–39 ms).
- Caveat vs the paper table: the paper's baseline (17.6/26.8/42.7 ms) differs
  from today's (18.8/45.1/94.6) — toolchain/bench-region drift — so ratios are
  only comparable within one snapshot. `docs/plots/*` (msshape figures) still
  predate the re-tune and need regeneration from this data before resubmission.

## Committed-chunk representation: witness commit = range-check chunk commit (2026-07-22)

Picked up the "batched Hyrax opens" follow-up; a fresh span profile at the
re-tuned (k=9, T=2^64) showed the opens are ALREADY fully batched (one
interleaved sumcheck + one merged same-column IPA, ~354 ms of a 2.95 s
multiswap-2^13 prove) — the real duplicate was the *commits*: `wq_commit`
committed the 64-bit limb polynomial (712 ms) and `rc_chunk_commit` committed
the SAME data again as 16-bit chunks (777 ms of which ~570 ms was the w/q
chunks). Implemented instead:

- **`IntegerModPCS::commit` now commits the base-2^16 chunk decomposition**
  of the limb-split polynomial (layout identical to the range check's F
  batch: `index = limb_index·stride + c`). The limb polynomial is never
  committed. Chunks are < 2^16, so the small-scalar MSM fast path is
  legitimate for EVERY `log_t` — the old `is_small = log_t <= 64` gate is
  gone.
- **Limb evaluation claims fold to single chunk claims** at a public tensor
  point: the weight vector `[2^{16c}]_c` is product-structured over the
  chunk-index bits, so `f_limb(z) = α^{-1}·chunk(z ++ x_*)` with
  `x_b = u_b/(1+u_b)`, `u_b = 2^{16·2^b}` (`chunk_fold_point`; unit-tested
  against direct MLE evaluation). Requires `⌈log_t/16⌉` to be a power of two
  — `validate()` enforces it (log_t ∈ {16, 32, 64, 128} all qualify).
- **F batches in the shared range check reuse the input commitment**: no
  fresh chunk commitment, no value-reconstruction sumcheck (the chunk→limb
  relation is definitional), and the LogUp-GKR witness claims land directly
  on the input commitment. Only the `a_j`/`b_j` batches still commit fresh
  chunk polys and reconstruct. Proof shrinks accordingly (2 fewer Hyrax
  commitments + 2 fewer reconstruction sumchecks per SNARK).

Measured single-thread (same machine/toolchain, criterion):

| workload | prove before | prove after | verify |
|---|---|---|---|
| MultiSwap 2^13 (2048-bit) | 3.11 s | **2.43 s (−22%)** | 41.8 ms (unchanged) |
| ECDSA MSM 2^13 (256-bit) | 386 ms | **271 ms (−30%)** | 24.6 ms (unchanged) |

Span deltas (multiswap): `rc_chunk_commit` 777→211 ms, `rc_reconstr` 53→5 ms,
batched opens 354→291 ms (9 targets instead of 11), `wq_commit` 712→787 ms
(now the 16-bit chunk MSM — same cost as the old limb MSM, but it's the ONLY
commit of that data).

**Re-tuned (k, T) after the change — (T=2^64, k=9) still optimal.** Commit
cost is now T-independent (total chunk bits are fixed), but T still gates
`log_p` through the Partial Eval Norm bound `k + k·log_p + max(log_t, log_p)
< 256`, so wider limbs shrink `log_p` → s explodes → chain work dominates:

| MultiSwap 2^13, chunked commit | best k | commit+prove |
|---|---|---|
| T=2^32 | k=10 | 2696 ms |
| **T=2^64** | **k=9** | **2385 ms** |
| T=2^128 | k=7 | 2929 ms |

ECDSA sweep confirms k=9 (271 ms; basin k=8–10). `DEFAULT_K = 9` and bench
`LOG_T = 64` unchanged. The msshape plots/table and the native-overhead
numbers predate this change and need regeneration before resubmission.

### Follow-on: one-pass limb split + u64 scalar-cast fast path

`split_value_into_limbs` was `numlimb` bignum `div_rem`s per value (≈500k
divisions + ~1M small allocations per multiswap polynomial); at T = 2^64
the limbs are literally the u64 digits. Rewritten as a one-pass bit-window
extraction over `to_bytes_le()`, plus a u64 fast path in
`biguint_to_scalar` (values ≤ 64 bits skip the 512-bit uniform reduction —
every limb at T ≤ 2^64 qualifies). Measured single-thread (multiswap 2¹³):
`red_limb_split` 132→14 ms, `red_to_fq` 41→7 ms, `wq_commit` 787→~650 ms
(its internal limb split got cheap too). Criterion: multiswap prove
2.43 → 2.13 s (−12%), verify unchanged.

### Follow-on: a/b layers commit as chunks too (zero-pad claims)

The `a_j`/`b_j` layers were the last double commit (stacked full-width
value commit ~83 ms + range-check chunk commits ~211 ms of the same
values). Now each layer commits its `a` and `b` chunk decompositions
directly (2 chunk commitments per layer, always `is_small`); chain
identity claims fold through `chunk_fold_point` like the F claims.
`b_j`'s 15-chunk bound (237 bits) is not tensor-friendly, so the fold
uses the FULL 16-slot weight tensor and the padding slot is pinned to
zero by a `range_zpad` claim — a free random-point opening claim
(Schwartz–Zippel) squeezed after the commitments. This generalizes to
any chunk count, so the power-of-two `⌈log_t/16⌉` restriction on
`validate()` is REMOVED. `SharedRangeCheck` now carries no per-batch
proof data at all (every chunk oracle is its target's own commitment;
reconstruction sumchecks gone entirely). Criterion: multiswap prove
2.13 → 2.03 s (−5%), verify 40.1 ms.

Cumulative across the three same-day rounds (criterion, single-thread):

| workload | prove (start of day → now) | verify |
|---|---|---|
| MultiSwap 2^13 | 3.11 s → 2.43 → 2.13 → **2.03 s (−35%)** | 40.1 ms |
| ECDSA MSM 2^13 | 386 ms → 271 → **255 ms (−34%)** | 24.0 ms |

k=9 re-confirmed optimal on the ECDSA sweep after all changes. Current
multiswap span shape: `wq_commit` 673 (33%), `rc_logup_gkr` 479 (24%),
batched opens 297 (15%), chain build ~225 (11%), `ab_commit` 166 (8%),
reduction ~85 (4%), GKR chunk-value rebuild 52 (3%).

Remaining small deferred item: phase 1 builds the a/b chunk values and
the range check rebuilds the same u64 chunk vectors for the GKR witness
trees (~52 ms) — thread them through `RangeBatchInputs` to skip the
rebuild.

### Follow-on (2026-07-23): fixed-width I256 chain arithmetic

The chain partial evaluations ran on heap `BigInt` (~8.4M mult-adds with
per-op allocation, plus a full `poly_bigint.clone()` per chain), but the
Partial Eval Norm bound `2^k·P^k·max(T,P) ≤ (q−P)/2 < 2^255` — the very
bound `validate()` enforces — guarantees every intermediate fits a
signed 256-bit word. Added a stack-allocated sign-magnitude `I256`
(`Copy`, length-aware schoolbook mul, single-word divmod) and a
fixed-width `integer_partial_evaluate_top_k_i256`; the chain build uses
it whenever `log_p ≤ 63` (the prime and reduced coordinates then fit
u64; the `BigInt` path remains as fallback), with a differential test
against the `BigInt` path on mixed-sign ~190-bit values.

Measured single-thread: `chain_build` 224 → 43 ms (5×); criterion
multiswap 2¹³ prove **2.03 → 1.87 s (−8%)**, ECDSA MSM 2¹³ **255 →
227 ms**; verify unchanged; Zinc+ prove gap now **~1.7×**. (The
reduction's `integer_mle_evaluate` stays `BigInt` — its chi factors are
full ~128-bit point coordinates over all 18 variables, unbounded by the
norm bound.)

### Follow-on (2026-07-23): breakdown scrutiny — three constant-factor fixes

A line-by-line audit of the 1.87 s profile against first-principles cost
models found three soft constants (everything else runs at its
arithmetic floor — see the justification table in
`zincplus_comparison.md`):

1. **MSM window sizing for short scalars.** `msm_small_rest`'s window
   heuristic `c = 0.69·log2(n) + 2` is tuned for full-width scalars; for
   16-bit chunks on 2048-point Hyrax rows it picked c=9 (2 windows, 511
   buckets each), making bucket aggregation ~⅓ of all adds. Shrinking
   `c` to the smallest value with the same window count (c=8, 255
   buckets) costs no extra data passes: `wq_commit` 686 → ~590 ms.
2. **Single-word chunk building.** `build_chunk_poly` went through
   `to_bytes_le()` + a Vec per limb; a T ≤ 2^64 limb is one u64 digit
   whose chunks are three shifts.
3. **Chunk→scalar table.** `t256::Scalar::from(u64)` is a Montgomery
   multiplication (~15 ns); the chunk pipelines convert ~4.7M sub-2^16
   values per proof (~70 ms) that take only 65,536 distinct values. A
   `OnceLock` table converts each once: GKR witness prep
   (`rc_chunk_commit` span) 38 → 4 ms.

Criterion: multiswap 2¹³ prove **1.87 → 1.71 s (−8.6%)**, verify
unchanged (42 ms); ECDSA MSM ~flat (226 ms — its MSM share is 8×
smaller). Zinc+ prove gap now **~1.55×**.

### Follow-on (2026-07-23): vartime bucket adds in `msm_small_rest`

Microbenchmarks (the `msm_op_cost_microbench` ignored test: mixed-add
288 ns — throughput-bound, not latency —, base-field mul 16.5 ns,
inversion 1.2 µs) exposed that `msm_small_rest`'s buckets used the
COMPLETE projective addition (~17 field ops, 288 ns) while `msm_10`
already had a vartime mixed-add `Bucket` enum (7M+3S ≈ 165 ns; safe —
commitment-key generators are never the identity). They also showed the
witness's chunk sparsity (~44% nonzero: bit columns, padding) means the
MSM does ~0.9 adds/point, so the per-add constant IS the cost. Swapping
`msm_small_rest` to the same `Bucket`: `wq_commit` 558 → 384 ms,
`ab_commit` 103 ms; criterion multiswap prove **1.71 → 1.47 s (−14%)**,
ECDSA **226 → 212 ms**, verify unchanged. Zinc+ prove gap **~1.33×**.

Remaining MSM lever, quantified by the same microbenches: batch-affine
bucket accumulation (~6 field ops + amortized 1.2 µs inversion ≈
~110 ns effective incl. ~12% conflict spill at 255-bucket granularity)
over the now-165 ns vartime adds → ~1.5× on the ~340 ms of remaining
data adds ≈ ~110–120 ms of prove. Needs the halo2curves-style
scheduler (conflict stamps, doubling/cancellation edge cases) — real
but bounded complexity.

### Lockstep (batched) LogUp-GKR — the MT bottleneck fix, landed (2026-07-23)

The deferred "batch the per-witness GKRs" fix, implemented as a
**lockstep restructure**: `gkr_prove_multi` absorbs ALL trees' roots
before any challenge, then advances every still-active tree through
layers/rounds together — per round, every active tree's round
polynomial is absorbed before the ONE shared challenge is squeezed.
Soundness per tree is the single-tree argument verbatim; per-tree proof
objects are structurally unchanged (only the transcript interleaving
differs, plus all roots move up front). Trees of different depths all
start at layer 0 and exit after their own last layer with leaf claims
at the then-current shared point — so equal-depth trees (the w/q chunk
trees) now get IDENTICAL leaf points, which the combined batch open
groups into shared weight passes. Bonus fixes in the same change: the
leaf denominators `r + w` are built from an incremental table
(2^16 adds) instead of a Montgomery multiplication per leaf (~55 ms),
the all-ones numerator leaf tables are never allocated, and the
verifier computes one eq evaluation per layer instead of one per tree.

Measured (multiswap 2¹³):
- **Single-thread (criterion): prove 1.47 → 1.37 s (−7%), verify
  42 → 39.4 ms.** ECDSA 212 → 188.7 ms / 21.8 ms.
- **Multithreaded (~14 cores): `rc_logup_gkr` 460 → 143 ms (3.2×
  scaling — the serial per-tree loop capped it at ~1.45×);
  prove-proper ≈ 530 ms.** ECDSA MT verify 13.5 ms.

The old serial-loop analysis in the "Multi-threading" section above is
superseded by this change.

Follow-on memory pass (same day): layer tables MOVE out of the level
build (`mem::take` + `split_off`) instead of two full copies each, and
`eval_cubic`'s two per-call field inversions (~2.4 µs × ~1500 calls on
EACH side) hoisted into per-walk constants. ST prove 1.37 → 1.36 s,
GKR span 460 → 427 ms, MT GKR 143 → 126 ms, verify 39.4 → 38.4 ms.
Remaining GKR gap vs the pure-mult floor (~200 ms) is the level build's
materialization and bind-write traffic — further reduction needs
build/round fusion (diminishing returns).

### Regenerated fig:nativeoverhead — post-campaign (2026-07-24)

Fresh single-threaded criterion pairs (rested machine) after the full
optimization series — every point now BEATS the paper's original ~9×
claim, and the super-linear 2¹⁴ anomaly is largely gone:

| cons | imod prove | native prove | overhead (was) | imod verify | native verify |
|---|---|---|---|---|---|
| 2^10 | 117.9 ms | 18.7 ms | **6.3×** (10.8×) | 25.8 ms | 14.4 ms |
| 2^12 | 257.6 ms | 44.6 ms | **5.8×** (12.5×) | 29.1 ms | 14.8 ms |
| 2^14 | 786.8 ms | 95.0 ms | **8.3×** (21.1×) | 33.1 ms | 16.7 ms |

Verify overhead ~1.8–2.0×. The `e9e9be5` recovery chase is moot: main
now exceeds the lost version's figures. `docs/plots/*` regenerated from
this data (annotations 6.3×/5.8×/8.3×).

### (k, T) re-sweep after the optimization series (2026-07-23)

The k=9/T=2^64 defaults were tuned when chains cost 224 ms and commits
~1.5 s; after the series (chains 5× cheaper, commits ~2.3× cheaper,
`is_small` no longer T-dependent) the sweep was re-run. Result: the
optimum is ROBUST — T=2^64 still beats T=2^128 (best k=7: 1797 ms vs
1315), and k=9–11 is now a flat basin within run-to-run noise
(1.27–1.34 s commit+prove; k≥12 clearly loses as s explodes). Keeping
k=9: the basin's larger-k end pays s=21–30 chains (bigger proofs, more
verifier chain work) for ≤3% prove inside the noise band. The
parameter choice surviving a 2.3× cost-model shift is worth noting in
the paper.

## T (limb/norm bound) coverage — complete grid (2026-07-14)

Closing the "is T=2^64 optimal everywhere?" question with measurements at
every cell (single-threaded, KSWEEP harnesses, best k per cell):

| config | T=2^16 | T=2^32 | **T=2^64** | T=2^128 |
|---|---|---|---|---|
| msshape 2^11–2^14 (256-bit) | dominated | dominated | **best (k=9)** | — |
| msshape 2^13 (256-bit) | 564 | 608 | **555 (k=9)** | 557 (k=6, tie) |
| msshape 2^15 (256-bit) | — | 2159 | **1964 (k=9)** | 2016 (k=6) |
| MultiSwap 2^13 (2048-bit) | 3540 | 3437 | **3053 (k=9/11)** | 3240 (k=7) |

**T=2^64 is optimal or tied at every measured cell.** Notes:
- T=2^128 at 256-bit is a near-tie (k≈6): halving numlimb again nearly pays
  for losing the `is_small` u64 fast-path MSM (limbs > 64 bits use the
  full-width path). At 2048-bit it clearly loses (+6%) — the wider witness
  makes the slow-path commit costlier. No reason to prefer it anywhere.
- T=2^64 is also the natural boundary: the max width whose limbs fill u64
  exactly and legitimately claim `is_small=true` (see the gate at
  `integer_modpcs.rs` f_limb commit and the >64-bit truncation regression
  test).
- At T=2^128 the optimal k drops to ~6–7 (s explodes beyond: k=9 → s≈93–98).

## Brakedown parameter sweep (2026-08-03)

Question: is the Bd prover fixable with parameters alone? Swept (k, T)
and the GLSTW code spec on multiswap 2^13 ST (temp env hooks BD_K /
BD_LOGT / BD_SPEC, since removed; re-add to the BDPCS bench block to
reproduce).

(k, T) at spec 5 — the Hyrax-tuned (k=9, T=2^64) is NOT optimal here:

| k | T | total | proof |
|---|---|-------|-------|
| 7 | 2^64 | 3.61 s | 28.90 MB |
| 8 | 2^64 | 3.18 s | 24.92 MB |
| 9 | 2^64 | 2.09 s | 21.75 MB |
| 11 | 2^64 | **1.99 s** | **20.08 MB** |
| 12 | 2^64 | 2.14 s | 20.07 MB |
| 9 | 2^128 | 2.90 s | 27.76 MB |

k=11 is a strict Pareto improvement for the Bd backend (fewer chain
layers -> fewer trees + opens; open cost is linear here, unlike Hyrax
where the MSM regime put the optimum at k=9).

Code spec at k=11 — trades commit(encode) time against query count:

| spec | commit | total | verify | proof |
|------|--------|-------|--------|-------|
| 0 | 592 ms | 1.74 s | 247 ms | 39.20 MB |
| 1 | 577 ms | **1.65 s** | 151 ms | 28.09 MB |
| 3 | 687 ms | 1.83 s | 157 ms | 22.59 MB |
| 5 | 855 ms | 1.97 s | 172 ms | 20.08 MB |

Prove-proper is flat (~1.07-1.15 s) across specs — only the encode
moves. Best-time config (spec 1, k=11): 1.65 s total, -21% vs the
shipped default, at +40% proof. Balanced pick (spec 5, k=11): 1.97 s /
20.08 MB / 172 ms — dominates the current default on every axis.
Conclusion: parameters recover 6-21%, floor ~1.65 s; the ~16x
field-width tax stays structural (integer-native code = future work).

T (limb bound) at spec 5 — 2^64 stays optimal under Brakedown:

| T | k | total | proof |
|---|---|-------|-------|
| 2^32 | 11 | 2.32 s | 21.73 MB |
| 2^32 | 12 | 2.13 s | 20.07 MB |
| 2^32 | 13 | 2.18 s | 18.59 MB |
| 2^16 | 13 | 4.29 s | 23.83 MB |
| 2^64 | 11 | **1.99 s** | 20.08 MB |

Smaller T loosens the norm bound (admits k=13, smallest proof seen at
18.59 MB) but loses on time; T=2^16 roughly doubles commit (per-limb
padding + limb-var overhead). Note k=11 at T=2^64 runs with log_p=16
(norm bound: k + k*log_p + max(log_t, log_p) < log_q), i.e. a much
larger s than k=9's log_p=24 — and still wins, so the s-repetition
cost is evidently not a dominant term. Relevant for the smaller-field
question (see zincplus_comparison.md).

## DynPrime<2> carrier for the sampled prime p (2026-08-03)

The transcript-sampled sumcheck prime p is 128 bits but was carried in
a 4-limb (256-bit) `DynPrime<4>` Montgomery form. Swapped the protocol
scalar to `DynPrime<2>` (microbench: mul 19.4 -> 10.9 ns, add 3.8 ->
2.2 ns, ~1.75x per op; all 194 tests pass).

Measured effect on multiswap 2^13 prove: **none** (A/B: 1.314 s 4-limb
vs 1.333 s 2-limb, within noise). Span diff explains it: the hot spans
(rc_logup_gkr 415 ms, bo_interleaved_sc 142 ms, chain/ab/opens) all run
over `t256::Scalar` -- the fixed 256-bit q-side PCS field -- and moved
0 ms. The p-side (mod-p Spartan sumchecks over DynPrime) is only ~10 ms
at 2^13; only the reduction/conversion spans improved (~30 ms total).

Takeaways: (a) the p-side carrier swap is kept -- strictly correct,
halves p-side element bytes, and the p-side cost grows linearly with
rows so it should matter at larger instances; (b) the real
smaller-field lever is the q side: the whole ~950 ms Mod-PCS open runs
in the PCS field, so the Brakedown 192-bit-q plan (fixed Solinas-style
prime, fast reduction) attacks the entire open, not just commit
hashing. Hyrax cannot shrink q (curve-pinned).

## Range check: all-zero block dropping (2026-08-04)

The committed chunk oracles carry large all-zero regions (41% padded
rows at multiswap 2^13, padded polys, zero limb slots). The shared
range check now splits every chunk polynomial into dyadic blocks of
2^RC_BLOCK_LOG (= 2^16) slots and feeds only blocks containing a
nonzero chunk into the LogUp multiset. Each dropped block is pinned to
zero by ONE fresh random-point opening claim (the `range_zpad`
Schwartz-Zippel technique generalized) -- strictly stronger than range
membership and O(block) to discharge thanks to `batch_weight`'s
existing boolean-head fast path. The active map travels in the proof
as untrusted advice, absorbed before any challenge: nonzero-marked-
inactive fails its zero claim; zero-marked-active wastes prover work.
Multiplicities count active entries only.

Measured (multiswap 2^13 ST): prove 1.333 -> **1.219 s** (-114 ms);
verify unchanged ~36.8 ms. Commit-inclusive vs Zinc+: 1.57 s vs
1.13 s -> gap 1.49x -> **1.39x**. Test:
`rc_zero_blocks_dropped_and_bitmap_pinned` (roundtrip at 8 blocks +
both bitmap forgeries rejected).

Next in this campaign: GKR round/build fusion (~100-150 ms), then
per-segment value bounds (bit rows pay 1 chunk slot instead of 128;
shrinks commit + range check + opens ~2.5-3x on multiswap).

## GKR bind/eval round fusion: tested, neutral, reverted (2026-08-05)

Implemented the fused round kernel (bind by `ri` + next round's
`[h(0), h(inf)]` sums in one table traversal, transcript-identical;
round 0 standalone, later rounds fed by the previous bind). All tests
passed. Same-thermal-state A/B at multiswap 2^13 ST: unfused 1.247 s,
fused 1.244 s -- **neutral**, so reverted.

Why the followups estimate (~100-150 ms) no longer holds: the
zero-block split caps every witness tree at 2^RC_BLOCK_LOG = 2^16
leaves, so per-round layer tables are cache-resident and the second
traversal is nearly free. The estimate predated blocking. Do not
retry unless RC_BLOCK_LOG grows or trees leave cache again; the
remaining range-check lever is per-segment value bounds (layout), not
kernel micro-optimization.

## Large matrix constants: accounting obligations (2026-08-07)

The circuits carry matrix coefficients far above T = 2^64 (the
conditional-multiply rows' `g−1` at ~2048 bits; reconstruction rows'
powers of two up to 2^351). This is sound TODAY because coefficients
are public shape data — never committed, never limb-split (no Spark:
the verifier evaluates A/B/C MLEs directly, ~1-2 ms at 2^13) — and
evaluated LC values stay small (big constants only multiply bits).
Two standing obligations:

1. **Paper soundness statement**: the main-relation fingerprint bound
   B_total must include coefficient norms. Adversarial row magnitude
   reaches ~2^6100 (2048-bit coefficient x range-checked 2^2048
   witness products) -> up to ~48 of ~2^121 candidate 128-bit primes
   divide a cheating row: soundness error ~2^-114 (vs ~2^-118 for 0/1
   coefficients). Harmless at a 128-bit sampled p; would bite below
   ~90 bits.

2. **If Spark is ever added** (sublinear verify at large nnz): the
   committed `val` table would hold 2048-bit coefficients needing
   limb-split + range check at log_t_f = 2048. Mitigations available:
   reconstruction coefficients are powers of two (closed-form MLE, no
   commitment needed); `g−1` is 4 distinct structured values ->
   hybrid Spark-plus-special-casing avoids committing wide values.

## Open-source readiness checklist (2026-08-13)

State of the repo audited for a public release. Code hygiene is
already good (fmt/clippy/typos gates green, no personal or tooling
traces in tracked files); the work is identity/metadata, the
uncommitted pile, and a history decision. Rough total: 2-4 focused
days, mostly decisions rather than code.

**Must do before flipping public:**

1. **Cargo.toml identity**: still claims `name = "spartan2"`,
   `version = 0.8.0`, `authors = [Srinath Setty]`,
   `repository = Microsoft/Spartan2`. Rename the crate, set real
   authors/repository, keep the MIT LICENSE (Microsoft copyright
   notice must stay; add our own line above it).
2. **Commit or drop the working tree** (~20 modified files + the
   `p3_adapter.rs` spike + `examples/commit_overhead.rs`). The p3
   spike is self-labeled "delete or promote after Phase A" and adds
   two public deps (`p3-field`, `p3-goldilocks`) — decide before it
   lands; check it doesn't break the CI wasm32 build job.
3. **Commit-history + repo-home decision**: decide whether the
   public repo keeps full history or starts from a squashed/grafted
   release commit, and which account/org hosts it; delete stale
   remote branches first.
4. **SUPPORT.md** is Microsoft boilerplate — rewrite or delete;
   likewise sweep for other upstream community files.
5. **README reproduce-the-paper section**: map each paper
   table/figure to its exact bench command + env
   (`RAYON_NUM_THREADS=1`, `target-cpu=native`), so the quoted
   numbers reproduce from a clean clone.

**Nice to have:**

- Retire the stale "Phase 2 step N" TODO markers and file-top
  `#![allow(dead_code)]` in `sumcheck_modp.rs`, `dyn_prime.rs`,
  `polys_modp/mod.rs`, `integer_modpcs.rs`, `ecdsa_msm.rs` (1-2 h).
- Decide the fate of docs/: ~3.2k lines of candid working notes
  (perf logs, retractions, plans). They are clean and arguably good
  research-log transparency, but keeping them should be a choice,
  not an accident.
- `tests/param_sweep.rs` / `tests/wide_value_probe.rs` read as
  probes, not tests — rename, gate behind `#[ignore]`, or move.
- State prominently that this is a research prototype (162
  unwrap/panic sites in the three core protocol files; no
  constant-time discipline claims).

**Timing**: do NOT gate the release on more PCS backends or more
benchmarks — those can land publicly afterwards. The only real
gates are the identity/metadata fixes and settling the working
tree; everything else is incremental polish an early release does
not preclude.

## Batch-affine msm_small: null result (2026-08-14)

Implemented batch-affine bucket accumulation for `msm_small_rest`
(raw-coordinate affine adds, Montgomery-batched inversions, buckets
lifted to checked points once at aggregation; doubling/cancellation
edges handled). Correct (203 tests + adversarial doubling/cancel
cases) but MEASURED SLOWER on the target workload: msshape c2^14
one-shot single-thread `wq_commit` 324 -> 365 ms (+13%), ab_commit
also up. (One-shot numbers, ±~10%; the confirming baseline rerun was
not completed.)

Why the win doesn't materialize: the Montgomery trick itself costs
~3M per element, so batch-affine is ~5.8M-equiv per add vs mixed
Jacobian's ~9.4M — only ~1.6x headroom — and the per-row MSMs are
short (~2k elements, ~350 rows at c2^14), so staging copies (72 B per
point per window), collision deferral, and per-window vectors eat the
margin. Library-class batch-affine wins come on million-point MSMs.

Reverted; implementation preserved off-tree for reference. Retry only
with one of: (a) cross-row scheduling — one batch stream spanning all
~350 row-MSMs of a commit so batches fill and the inversion amortizes
globally; (b) precomputed per-generator window tables (commitment key
is fixed; memory-bound, ~512 MB at 4-bit windows for 2^19 gens); or
(c) row lengths >= 2^15. The commit-time gap vs native (5.5x at
msshape c2^14: 324 vs 59 ms) remains open; next candidate is the
aspect-ratio lever (longer rows amortize buckets AND raise the
batch-affine ceiling — the two compose).

## 128-bit field microbench: q=128 branch is GO (2026-08-19)

Four-way mul microbench (`field_128_candidates_microbench` in
dyn_prime.rs, --ignored; single-thread, target-cpu=native, M4 Pro),
all 128-bit candidates over M127 = 2^127 − 1:

| candidate | latency ns (vs t256) | throughput ns (vs t256) |
|---|---|---|
| t256 (4-limb, today) | 14.44 | 8.38 |
| DynPrime<2> (runtime modulus) | 7.52 (1.9x) | 3.53 (2.4x) |
| ff_derive F127 (compile-time Montgomery) | 3.60 (4.0x) | 2.75 (3.1x) |
| hand M127 (u128 + Mersenne fold) | 3.76 (3.8x) | 1.72 (4.9x) |

The >=3x go/no-go gate for the q=128 / hash-based-PCS operating point
CLEARS: ff_derive alone (10 lines, zkcrypto-audited codegen, ff-trait
native) hits 3-4x; the hand-rolled Mersenne fold hits 4.9x on
throughput (the metric that matters — binds/evals are
independent-slot loops). DynPrime's runtime-modulus tax is ~2x vs
compile-time at equal limb count, confirming it was never the right
performance vehicle.

M127 is uniquely available to us: two-adicity 1 (no FFTs — fatal for
FRI/STARK stacks, irrelevant for our sumcheck+expander-code stack).
Norm-bound grid at log_q=127 ~ the 128 row: k=5-6, log_p 17-20,
s=16-25, ~0.4-0.5x commit overhead; challenges from F_{q^2} = 2^254.

Next steps for the branch (est. ~1 month total): promote a fixed
M127/F127 field (start from ff_derive; specialize hot paths to the
Mersenne fold where profiles say so) -> wire as ModEngine q-side ->
port Brakedown with base-field data / extension-field coins ->
head-to-head vs Zinc+ at the fast-prover operating point.

## Brakedown commit over F127: measured 1.8-1.9x (2026-08-19)

`brakedown_field_ab` (ignored test, brakedown/mod.rs): the generic
commit path instantiated over ff_derive M127 vs t256, equal element
count (16-bit chunk data, single-thread, native). Total commit
1.75-1.92x faster; BDSPLIT decomposition at 2^20: encode 353.5 ->
177.8 ms (2.0x, 82% of commit), column hash 64.8 -> 35.7 ms (1.8x,
half the bytes), Merkle tree ~unchanged (field-blind). Encode lands
at 2x rather than the mul microbench's 3-4x because the expander
encode is add- and bandwidth-heavy, not mul-bound.

Implications: (a) the commit layer of the q=128 operating point is
real and needed no backend changes (PrimeFieldExt was already
generic; F127 needed only a 10-line derive + from_uniform via
2^128 = 2 mod M127); (b) net commit gain in the full q=127 design
after the ~1.45x aux-volume tax: ~1.3x vs Brakedown-t256 at equal
witness — the bigger q-side wins (GKR, chains, opens at 3-5x ops)
still require the ModEngine reparameterization port; (c) next encode
lever if it matters: unrolled/SIMD Mersenne-fold in the encode inner
loop (the hand-M127 kernel's 4.9x throughput suggests headroom over
ff_derive's Montgomery in exactly this loop shape).

## Challenge-soundness target lowered to 117 bits (2026-08-19)

`LAMBDA_BOUND2 = 117` (new constant, integer_modpcs.rs) now drives the
Soundness Bound 2 check instead of the hardcoded λ = 128, by explicit
decision: overall system soundness is already bounded by the ~2^-114
fingerprint prime-sampling term, so full 128-bit challenge soundness
over-secured one term. Consequence: a ~2^127 base field (M127) passes
Bound 2 with challenges drawn from the BASE field — no extension
field needed anywhere in the q=127 design, which removes an entire
work item (dual-field arithmetic) from the Brakedown/small-field
port. Bound 1 (prime-count) and the value-magnitude bound still
target λ = 128. May be lowered further if a future instantiation
needs it. Paper obligation when the small-field instantiation ships:
state achieved soundness as the min (~114-117 bits), as already done
for the prime-sampling term.

## Phase-1 field-genericization: state + remaining plan (2026-08-19)

Done (each step green + committed): CommitBackend has `type Scalar`
(both backends pin t256 for now); OpenTarget + trait methods generic;
combined batch open (prove/verify), absorb_batch_claims,
batch_weight, OpenClaims, CombinedBatchOpen all field-generic with
byte-level transcripts (bo_lambda/cbo_c squeezes now
from_uniform(squeeze_bytes) — transcript bytes changed, both sides
together). Six pins remain: prove_one_poly, verify_one_poly,
finish_batch_open, finish_batch_verify, prove/verify_shared_range_check.

Remaining plan:
1. Chains cluster: parameterize SmallPrimeOpening / IterationOracles /
   ChainData + mle_evaluate_fq, convert prove/verify_one_poly.
2. Range-check cluster: re-parameterize logup_gkr from `E: Engine` to
   `E: SumcheckEngine` (+ `Scalar: PrimeField` bound — SumcheckEngine
   already exists with standalone impls and an Engine blanket impl);
   give CommitBackend a `type SE: SumcheckEngine<Scalar = Self::Scalar>`
   so SharedRangeCheck stays single-parameter; Hyrax sets SE =
   T256HyraxEngine, a future M127 backend mints a tiny curve-free
   engine struct.
3. Finishers: finish_batch_open/verify unpin once 1+2 land; then
   LOG_Q moves from a module const to a B::Scalar-derived value
   (params-driven), and the derive()/validate() formulas take it as
   input.

## Phase-1 function genericization COMPLETE (2026-08-19)

Zero pinned protocol functions remain in integer_modpcs.rs; the only
t256 binding left is HyBackend's own `type Scalar` declaration.
Nine green commits, each gated (tests/clippy/fmt/typos). Structure:
CommitBackend carries `type Scalar` + `type SE: SumcheckEngine`;
logup_gkr and sumcheck.rs are SumcheckEngine-bound (zk methods split
into an Engine impl); all protocol structs parameterized (t256
defaults so callers didn't churn); transcripts byte-level or B::SE::TE.

Remaining for a full field swap:
1. **LOG_Q -> field-derived.** Still a module const (=256) used by
   derive()/validate() and the chain bounds; should become a
   B::Scalar-derived value (field_q::<F>().bits()) threaded through
   IntEvalParams.
2. **MontgomeryLimbs is 4-limb-hardcoded** ([u64; 4]); the sumcheck
   hot kernels require it (explicit where-clauses). An M127/F127 field
   needs the trait generalized over limb count or a padded 4-limb impl
   (correct but wasteful; generalizing is the right fix and also
   speeds the 2-limb field's delayed reduction).
3. **Driver level**: imod_spartan_modp.rs still names t256 engines
   (~13 sites) - the top-level SNARK types, not protocol internals.
4. Then: F127 SumcheckField/TranscriptRepr/PrimeFieldExt impls + a
   curve-free SE struct + a Brakedown backend declaration at
   Scalar = F127 = the end-to-end smoke test.

## Smoke-test checklist for the M127/Brakedown stack (2026-08-20)

Phase-1 plumbing is COMPLETE through this point: protocol functions
field-generic, log_q in IntEvalParams (derive_for_q), hot-kernel
bounds on DelayedReduction (not 4-limb MontgomeryLimbs). What remains
is declaration work, mirroring the existing T256DynPrimeBdEngine
pattern:

1. F127 field module: ff_derive M127 + PrimeFieldExt +
   TranscriptReprTrait + an eager DelayedReduction impl (correctness
   first; a WideLimbs<5> 2-limb fast path later). SumcheckField
   arrives via the blanket impl.
2. F127Engine: 4-line SumcheckEngine (Scalar = F127, TE = Keccak).
3. Bd127Backend: CommitBackend at Scalar = F127, SE = F127Engine —
   copy BdBackend (~80 lines; its param/data caches are t256-typed
   statics, so a parallel impl not a generic one).
4. M127DynPrimeBdEngine: ModEngine mirroring T256DynPrimeBdEngine +
   an IntegerModPCS instantiation at Bd127Backend + a driver
   setup impl (~30 lines). CHECK: the IntegerModPCSBd struct layer
   (BdModCommitmentKey etc.) may still be t256-typed internally.
5. Params: derive_for_q(127, ...) — LAMBDA_BOUND2=117 already admits
   a 127-bit field (127 >= 117 + log2(s*n) for our shapes).
6. The smoke test: small Mod-R1CS instance, prove+verify end-to-end
   over the M127 stack. Fat proofs, unoptimized — validates the
   pipeline; perf work (2-limb delayed reduction, two-tree opening)
   comes after.

## M127 first real-scale run: chain-bound slack bug found (2026-08-20)

The M127/Brakedown MultiSwap one-shot (M127=1 bench block; params
derive_for_q(127, 2048, 16, k=5, 13) -> log_p=20 s=16) fails in the
range check: "witness 216 value 126975 >= 2^16" — a b-side chain
value's chunk spilled the 16-bit table. Setup, witness commit, and
the chain build all ran; the failure is a NORM-BOUND issue, not
plumbing: `log_bound_b = log_q - log_p + 1` (integer_modpcs, two
sites, no derivation comment) under-budgets the b-side chain values.
Back-derivation from the spilled chunk puts the real magnitude near
2^112-113 vs the formula's 2^107 budget at these params — consistent
with a missing k-dependent term (2^k interpolation-sum factor in the
chain identity). At q=256 the ~130 bits of slack masked this
entirely; the toy M127 smoke test passed because its values are tiny.

Next steps (in order):
1. Instrument: log max |b_j_shifted| bit-length per layer in a q=127
   run to measure the true bound empirically.
2. Re-derive log_bound_b from the paper's Partial Evaluation analysis
   for general (q, p, k); fix the formula; check whether the q=256
   configuration's stated bounds also need the correction on paper
   even though padding absorbed it in practice.
3. Re-run the M127 MultiSwap one-shot.

The toy roundtrip (imod_modp_m127_toy_roundtrip) stays green; this is
exactly the class of bug the real-scale smoke run exists to catch.

## CORRECTION + first M127 numbers (2026-08-20)

The "chain-bound slack bug" section above is RETRACTED: the paper's
b-side bound (||g|| < (q-P)/2, |b| < q/P — documented verbatim at
shift_b()) and its k-aware derivation were correct all along. The
MultiSwap failure had two implementation causes, both from this
week's refactors: (1) the LOG_Q parameterization pass left shift_b()
computing q/P from the t256 modulus (fixed: shift_b::<F> generic over
the actual q-side field; the CHAIN_BITS instrumentation showed
236-bit b values against a 108-bit budget = exactly the 256-bit q);
(2) F127::from_uniform walked LE 128-bit chunks lowest-first with
highest weight, embedding every >64-bit value as 8x itself (fixed:
.rev(); caught by the reconstruction sumcheck at verify).

First end-to-end M127/Brakedown MultiSwap 2^13 (single-thread,
unoptimized): **commit+prove 9.46 s, verify 272.6 ms, proof
51.2 MB** (per_poly 9.8 KB, range_check 374 KB, combined_open
50.8 MB). vs Hyrax/t256: 1.32 s / 21 ms / 175 KB. Attribution of the
gap, in order: (a) the unbatched per-target Brakedown openings — the
wq_open span alone is ~8.0 of 9.46 s and combined_open is 99% of the
proof; this is exactly what the phase-2 two-tree batched opening
removes; (b) eager DelayedReduction on F127 (every product reduced;
the WideLimbs<5> fast path pending); (c) 16-bit limbs quadruple the
limb-slot count vs log_t=64. The range check at 374 KB (vs 89 KB on
t256+Hyrax at coarser limbs) says the protocol core is healthy;
the floor is real and the levers are known.

## Two-tree batched opening: commit schedule VERIFIED (2026-08-20)

Traced every transcript absorb/squeeze through the Bd prove path.
Actual order per polynomial (prove_one_poly): absorb int_v' +
reduction rounds -> sample the s primes -> compute ALL chain layers ->
commit ab chunk polys (absorbed) -> squeeze gammas -> absorb claim
evals. Then finish_batch_open: range sub-transcript (absorbs all
chunk + mult comms, THEN LogUp r / zblk / zpad / rv), batch
sub-transcript (lambdas, cbo challenges). Confirms the user's
argument: the whole IntEval is computable and committable before ANY
checking challenge — gammas, range, and batch challenges all come
after, and nothing in the chain data depends on them.

One restructure needed: prove_batch currently runs prove_one_poly
per polynomial SEQUENTIALLY, so poly i+1's chain commits land after
poly i's gammas. The dependency analysis says this interleaving is
unnecessary; the code even anticipates the split (ChainProverState:
"collected in phase 1, consumed in phase 2"). Hoist: all polys'
phase-1 (reduction, primes, chain commits) first, then all gammas +
claims. Transcript ordering changes (self-consistent, both sides).

Tree membership then:
- Tree 1 (exists already): the instance's witness/quotient Brakedown
  commitments (could merge comm_w/comm_q into one root).
- Tree 2 (new): ALL ab chunk polys across layers/primes/polys + the
  range-check F chunk polys + the multiplicity table — everything the
  open creates, committed in one stacked matrix after prime sampling.
- Then every challenge, then the checks; opened via ~128 shared
  columns per tree. Aspect chosen for short columns (opening size),
  proof target ~1-1.5 MB at MultiSwap 2^13 (from 21.7 MB), verify
  ~50-70 ms (from ~190), prove -200-400 ms.

Note: the reduction sumcheck's rounds sit before prime sampling and
involve no commitments — unaffected by the restructure.

## Hash-mode prover at curve-mode parity (2026-08-20)

Two optimizations on the q=256 Brakedown mode, both measured on
MultiSwap 2^13 single-thread:
1. Input-sparsity guard in the encoder (mul_vec skips zero inputs):
   wq_commit 916 -> 581 ms.
2. Code-spec + k sweep on guarded code (BDSPEC/BDK knobs): k=11
   dominates k=9 everywhere; spec1 (beta=.0444, R=1.47) is the
   prover-time optimum. New defaults: spec1 + k=11.

Frontier (spec/k=11): spec1 1.35s/142ms/28.0MB; spec2 1.42/145/24.9;
spec3 1.43/141/22.5; spec5 1.58/162/20.0. Chosen point 1.41s
measured post-default (run variance ~4%).

Scoreboard: hash mode 1.41 s prove / 151 ms verify / 28 MB — prover
at PARITY with the curve mode (1.32 s), 1.5x ahead of Zinc+ (2.06 s)
with 3.4x their verify. Remaining prover levers: GKR uniskip (389 ms
range check), opening machinery (~290 ms). Proof-size ladder
unchanged (batching ~-30%, packing for the big cut).

## GKR uniskip step 1 landed: mul_u64_scaled (2026-08-20)

The prerequisite primitive from gkr_uniskip_plan.md: multiply a
Montgomery-form element by a plain u64 at one-fold cost, returning
a*b*2^-64 (uniform scale fixed once per accumulated sum via
from_u128(1<<64)). Differential-tested (2000 cases + edges).
Measured on M4 Pro, native codegen: latency 1.22x over the full
mult (ASM full mult is too fast for a chain win) but THROUGHPUT
3.60x (2.29 vs 8.23 ns) in the independent-slot accumulation shape
the skip round uses - the plan's >=3x gate passes where it counts.
Remaining: steps 2-5 (skip-round prover for witness-tree leaf
layers, verifier barycentric + proof-format change, differential
tests, tune ell in {3,4,5}). Payoff estimate at current numbers:
range-check GKR 389 ms -> ~260-300 ms, hash-mode total ~1.42 ->
~1.30-1.35 s.

## Prover race vs curve mode: 80 ms short; uniskip is the closer (2026-08-20)

Median-of-5 same-state one-shots: hash mode 1.429 s (tight:
1.425-1.442) vs curve mode 1.35 s. Landed today: compact
length-prefixed leaf serialization (hash 57 -> 41 ms per big poly;
floor is per-leaf blake3 overhead at 107k tiny leaves, not bytes;
transcript bytes change benignly). Remaining lever: the GKR
univariate skip (301 ms range-check GKR, primitive validated at
3.60x throughput).

Design note for the skip in the CURRENT (post gamma-RLC lockstep)
GKR: the zero-block layout makes ALL range-check trees depth 16
(2^16-leaf blocks + the 2^16 table tree), so every tree reaches its
leaf layer at the SAME lockstep layer - the skip round synchronizes
across trees and gamma-RLC combines the per-tree evaluation-form
skip polynomials exactly like normal rounds. Leaf data is
small-value structured for every tree (witness trees: r + w with
w < 2^16, ones-numerators elided; table tree: r + j index-affine,
multiplicity counts). Skip ell in {3,4,5} of the 16 leaf rounds,
proof grows ~3*2^ell*32 B per layer (KB-scale). Projected: GKR
301 -> ~200-230 ms, total ~1.30-1.33 s vs curve 1.35 - clearing the
bar narrowly; combined with remaining opening trims, margin grows.

## RS-code PCS direction: field swap unlocked, NTT gated (2026-08-21)

t256's q-side field has two-adicity 1 - WHIR/STIR/FRI-family PCS are
structurally unavailable over it. BUT the phase-1 genericization
makes the q-side field a declaration, and bn254::Fr (two-adicity 28,
PrimeFieldExt already impl'd in-tree) satisfies the same norm bounds
at 254 bits. Gate measurement (ntt_encode_gate, dyn_prime.rs): naive
radix-2 NTT over bn254::Fr = 277 ms at 2^20 / 587 ms at 2^21
single-thread vs ~130 ms expander encode; optimized NTT plausibly at
parity, but RS rate blowup (x2-4) and the LOSS of the input-sparsity
guard (NTTs are dense) are real. Projection: WHIR-class swap =
proofs ~100-300 KB (/70), verify fast, prover +20-55% (~1.7-2.2 s) -
a THIRD Pareto point (balanced curve-free), not a dominator. Next
de-risk: harness-bench the WHIR reference implementation
(arkworks-based) at our sizes over bn254 before building - the
Garuda-dig methodology.

## Proof-size easy wins: 22.5 -> 17.1 MB raw (2026-08-21)

Three wire-format changes, no protocol or performance cost (1.39 s /
143 ms unchanged within variance):
1. Compact length-prefixed encoding for shipped column entries
   (mirrors the hashing encoding; deserializer enforces minimality,
   so no malleability) - the native version of the zstd win.
2. Column indices dropped from the wire: the verifier re-derives them
   from the transcript and was only checking the shipped ones.
3. BD_LAMBDA 128 -> 117, aligned with the accepted system floor
   (same argument as LAMBDA_BOUND2): ~9% fewer opened columns.
zstd now 14.9 MB (raw and compressed converging - the structure is
captured natively). Next size step remains stacking (~-4.5 MB ->
~12.5 raw), then the packing/PCS-swap fork.
