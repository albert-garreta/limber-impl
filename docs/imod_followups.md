# IntMod-Spartan deferred work

Running list of optimizations, hygiene items, and tests we've identified but
not yet implemented. Each entry links to its phase of origin and an estimated
size; promote into a phase plan when picked up.

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

## Phase 2 follow-ups (will accumulate here as we work through it)

(none yet)

## Phase 3 follow-ups (identified during step A–C implementation, 2026-06-01)

### Performance / structure

- **`IntEvalEvalArg` is huge under default params.** Step C produces, per
  prove call: `s` chains × `t` iterations × 2 polynomial Hyrax commits, plus
  `s × (3t + 1)` Hyrax openings (each carries `f_y`, `blind_eval`, and a
  Hyrax eval-argument). For default params (s≈10, t depends on n/k), this
  grows fast. **Fix candidates:**
  - Batched Hyrax openings: a single multi-point open per oracle instead of
    one Hyrax::prove per (i, j, oracle).
  - Re-randomize across primes: many of the openings are at the same γ-prefix
    but on different oracles — a single random-linear-combination open could
    replace 3 of them per iteration.
  - Compact the BigInt `int_v_prime` serialization (currently sign byte +
    8-byte LE length + LE magnitude; could be tighter).

- **Per-`p_i` matrix-style work is not parallelized.** The `s` chains are
  embarrassingly parallel — each is independent until γ is sampled. **Fix:**
  rayon-parallelize the per-chain phase 1 loop in `IntEvalModPCS::prove`.
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

- **Range check on commit is not implemented.** Phase-3 step D. Without it,
  IntEvalModPCS soundness depends on the application enforcing the witness
  bound. The non-negative witness assumption + the `T_f` parameter define
  the bound; we need a sumcheck-based bit-decomposition or lookup argument
  to enforce it.

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

- **`TrivialIntModPCS` and `BridgeModPCS` are now dead code.** Kept as
  reference. After Phase 3 is stable, delete.

- **The IntEval impl is concrete to `T256DynPrimeEngine`.** No abstraction
  over which underlying F-PCS to use. Once we have multiple `ModEngine`s,
  consider generalizing — but probably not before then.

- **No serialization tests for `IntEvalEvalArg`.** The struct includes
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
