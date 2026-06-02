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

- **`DynPrime<4>::from_bytes_reduce` truncates to 32 bytes.** Anything
  wider than 256 bits silently loses its top bytes. For our toy witnesses
  (≤ 64-bit) we're fine, but if `T_f` ever exceeds 256 bits we'd be
  silently producing wrong reductions. **Fix:** iterative chunk-wise reduction (or a
  wider intermediate `Uint`). Medium.

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

  Dominant Phase-2 costs (not yet attributed by profile): DynPrime
  arithmetic in the sumcheck (~3-5× per op vs t256::Scalar), `s`
  rejection-sampled small primes per Mod-PCS open (Miller-Rabin is
  measurable), and step-C identity opens (3 Hyrax opens per iteration per
  prime). All three are addressed elsewhere in this list.

  Caveat on the comparison: P1/P2 imod synthetic shape has 3 nnz/row
  (one per matrix); plain Spartan's bellpepper-synthesized multiplication
  uses denser matrices. Proof-size comparison is therefore not
  apples-to-apples; constraint count is the meaningful axis until we
  shape-match more carefully.

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

- **`IntEvalParams` defaults are baked into `IntEvalModPCS::setup` and
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

  3. **No application path to set params.** `IntEvalModPCS::setup_with_params`
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
