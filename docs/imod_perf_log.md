# imod_spartan_modp perf iteration log

Per-change timings on the canonical `imod_spartan_modp/prove/c2^10_v2^12`
benchmark (10 samples per run, criterion median). Each row is one
commit-sized change; record the section breakdown from the
`RUST_LOG=info` one-shot pass alongside the criterion topline so we know
which span moved.

Run command:

```
RUSTFLAGS="-C target-cpu=native" cargo bench --bench imod_spartan_modp -- "prove/c2\^10_v2\^12"
```

For the section spans, prefix with `RUST_LOG=info` and read the first
post-warmup `integer_modpcs_prove` (for `w` open) / second (for `q` open)
from the one-shot pass — criterion iterations interleave with them.

## Baseline: `428806f` (Revert PCS/sumcheck to c42ad8c baseline) — 2026-06-07

| metric              | value      |
|---------------------|------------|
| criterion median    | **236.31 ms** |
| criterion 95% CI    | 235.20 – 237.89 ms |
| SNARK one-shot      | 236 ms     |
| `w` IntegerModPCS::prove | 138 ms |
| `q` IntegerModPCS::prove | 94 ms  |

### `w` open section breakdown (138 ms)

| span                  | ms |
|-----------------------|----|
| imod_pcs_reduction_sc | 3  |
| imod_pcs_chain_phase1 | 2  |
| imod_pcs_chain_openings | 26 |
| imod_pcs_curr_batch   | 1  |
| imod_pcs_aprev_batch  | 16 |
| imod_pcs_rc_flimb     | 34 |
| imod_pcs_rc_ab        | 53 |

### Inside `rc_flimb` (34 ms, bit_poly n=2^17, log_bound=32)
- rc_bit_validity_sc: 3
- rc_bit_open_validity: 9 (commit 1 + IPA 4)
- rc_value_open: 8 (commit 3 + IPA 4)
- rc_bit_open_reconstr: 9 (commit 1 + IPA 4)

### Inside `rc_ab` (53 ms = a_j 18 ms + b_j 27 ms)
- a_j (n=2^14, log_bound=31): bit_validity_sc 1, bit_open_validity 8, value_open 1, bit_open_reconstr 8
- b_j (n=2^17, log_bound=227, stride=256): bit_validity_sc 3, bit_open_validity 11, value_open 1, bit_open_reconstr 11

## Drop `comm_eval` hiding from `hyrax_open_at` — 2026-06-07

Change: `hyrax_open_at` previously sampled a per-open `blind_eval` and
committed `comm_eval = Hyrax::commit(ck_eval, &[f_y], &blind_eval, false)`.
The IPA needs `comm_eval` but `f_y` is already sent in the clear, so
hiding adds nothing. Switched to a deterministic zero blind so
`comm_eval = G^{f_y}` is recoverable on the verifier from `f_y` alone;
dropped `blind_eval` from `SmallPrimeOpening`. Added
`HyraxBlind::zero(ck, n)` constructor.

Touches `src/provider/pcs/integer_modpcs.rs` (struct field, `hyrax_open_at`,
`hyrax_verify_open`, import), `src/provider/pcs/hyrax_pc.rs` (constructor).

| metric              | before     | after      | Δ        |
|---------------------|------------|------------|----------|
| criterion median    | 236.31 ms  | **235.32 ms** | −0.4%   |
| criterion 95% CI    | 235.20–237.89 | 234.37–236.67 | −     |
| `w` IntegerModPCS::prove | 138 ms | 135 ms | −3 ms    |
| `q` IntegerModPCS::prove | 94 ms  | 93 ms  | −1 ms    |

`change` line: `[-1.5473% -0.7497% +0.0593%] (p = 0.10 > 0.05)` — within
noise band, no statistically significant change to prove time. The
hypothesized 15% win was based on a wrong attribution: the `hyrax_prove_commit`
span I saw (1–3 ms per open) is the load-bearing `comm_LZ` MSM inside
`Hyrax::prove`, not the dead-weight `comm_eval` setup. The actual
`Hyrax::blind` + 1-elem MSM the change removes is sub-millisecond.

The change is still worth keeping for:
- proof-size reduction (drops one `HyraxBlind` scalar per opening ×
  ~30 openings per SNARK ≈ 960 bytes/proof at 2^10/2^12)
- API cleanup (no more fake hiding around an in-clear eval)
- removes one of the `comm_eval`/`blind_eval` dead-weight items from
  `imod_followups.md`

All 157 tests pass. Tests verified: 13 imod-specific + 144 others.

## k sweep on `prove/c2^10_v2^12` — 2026-06-10

`IntEvalParams::derive` adapts `(log_p, s)` to `(k, log_t, n)` but `k`
itself is a fixed input (`DEFAULT_K = 7`). Added an `IMOD_K` env
override to the bench (`setup_for` helper) plus a `tests/param_sweep.rs`
diagnostic that prints the derived params per `(k, n)`.

Derived params at `num_vars = 12` (w-poly), `log_t = 32`:

| k | log_p | s   | t | criterion prove median |
|---|-------|-----|---|------------------------|
| 4 | 50    | 4   | 2 | 332.59 ms (+40%)       |
| 5 | 41    | 6   | 2 | 287.94 ms (+21%)       |
| 6 | 35    | 7   | 1 | **230.41 ms (−2.8%)**  |
| 7 | 30    | 10  | 1 | 236.97 ms (baseline)   |
| 8 | 26    | 13  | 1 | **231.26 ms (−2.4%)**  |
| 9 | 23    | 19  | 1 | 248.42 ms (+4.8%)      |

Verify at k=6: 122.71 ms vs 126.12 ms (−2.7%).

Reading — the curve is a shallow bowl over k=6..8 with cliffs on both
sides, and the two cliffs have *different* causes:

- **Low-k cliff (t = 2) is mostly intrinsic, not batching.** The first
  iteration's `a_1`/`b_1` polys have size `2^(n−k)`, so shrinking `k`
  *grows* the committed bulk: total a/b range-check bits at k=4 are
  ~327k vs ~147k at k=7 (~2.2×), and per-chain committed entries grow
  the same way (sweep table). On top of that, `t = 2` adds the
  unbatched `s·(t−1)` `a_prev` opens and two more range-check groups —
  batching gaps, fixable — but even with perfect batching the bulk
  growth keeps low k behind at this shape.
- **High-k cliff (k ≥ 9) is mostly batching/overhead.** Bulk *shrinks*
  with k (k=8 has half the a/b range-check bits of k=7: s grows slower
  than `2^k` shrinks the polys), but the s-proportional unbatched
  overhead grows: more final opens (13 → 19), more chain bookkeeping,
  and `n_pad` rounding waste in the range-check stacks. This is the
  direction better batching (Thread B final-open stack + rc-value-open
  folding) would unlock.
- k=7's local bump vs k=6/k=8 is plausibly `n_pad` waste: s=10 pads to
  16 (37% padding rows) vs 7→8 (12%) and 13→16 (19%). Not isolated.
- The collaborator's `logup-gkr-range-check` branch tightens
  Soundness-1 (prime-counting bound), shrinking `s` at every `k` and
  shifting the optimum toward higher k (their MultiSwap data: k=10
  beats k=7 under the new bound). Re-sweep after merge; the `IMOD_K`
  knob + `tests/param_sweep.rs` are in place for it.

## Post-merge baseline: LogUp-GKR main (`cab16f5`) — 2026-06-11

`logup-gkr-range-check` merged to main (PR #1) plus six follow-on
commits: combined batch open (ALL batched opens → one interleaved
sumcheck + one same-column IPA, `2bc8fd2`/`6d06642`), GKR leaf-layer
specialization (`249276c`), Dao-Thaler eq split in the GKR layer
sumchecks (`cf1e385`, `cab16f5`). Local zero-blind `comm_eval` work
re-applied on top, extended to the new merged-open call sites.

| metric (criterion median, c2^10_v2^12) | pre-merge | post-merge | Δ |
|---|---|---|---|
| prove  | 236.97 ms | **192.24 ms** | **−18.9%** |
| verify | 126.12 ms | **37.95 ms**  | **−69.9%** |

168 lib tests pass (was 157; LogUp module added 11). fmt + clippy clean.

Param sweep under the new prime-counting Soundness-1 bound
(`num_vars = 12`, `log_t = 32`) — `s` shrinks at every `k` and the
valid range extends to k=16:

| k | log_p | s (old bound) | s (new bound) | t |
|---|-------|----------------|----------------|---|
| 6 | 35    | 7              | 6              | 1 |
| 7 | 30    | 10             | **7**          | 1 |
| 8 | 26    | 13             | 9              | 1 |
| 9 | 23    | 19             | 11             | 1 |
| 10| 21    | 26             | 13             | 1 |
| 11| 19    | 43             | 17             | 1 |

k=7's `s` drops 10 → 7 (n_pad 16 → 8 in any stacked structures, fewer
chains everywhere). A fresh `IMOD_K` sweep on the new main is the
obvious next measurement; with the combined batch open the per-chain
overhead is much flatter in `s`, so higher k may now win.

### Fine-grained prove breakdown (one-shot, c2^10_v2^12)

Re-added per-phase spans (rc_chunk_commit / rc_mult_commit /
rc_logup_gkr / rc_reconstr inside `imod_pcs_rc_shared`; bo_w_build /
bo_interleaved_sc / bo_merged_ipa inside `imod_pcs_batched_opens`) —
upstream's `01b6544` spans had been dropped by the `2bc8fd2`
restructure.

| span | w open | q open |
|---|---|---|
| imod_pcs_reduction (limb sumcheck) | 1-3 | 0 |
| imod_pcs_chain_phase1 | 2 | 1 |
| imod_pcs_chain_claims | 5 | 2 |
| rc_chunk_commit | 2 | 2 |
| rc_mult_commit | 1 | 1 |
| **rc_logup_gkr** | **48** | **36** |
| rc_reconstr | 3 | 2 |
| imod_pcs_rc_shared (total) | 56 | 41 |
| bo_w_build | 13 | 9 |
| bo_interleaved_sc | 8 | 6 |
| **bo_merged_ipa** | **20** | **19** |
| imod_pcs_batched_opens (total) | 41 | 34 |
| **integer_modpcs_prove** | **107** | **81** |

SNARK prove one-shot 193 ms ≈ criterion 197.4 ms. Verify: essentially
all of `integer_modpcs_verify` is `imod_pcs_verify_batched_opens`
(16-21 ms per open; verify_rc and verify_chains are 0 ms).

## MultiSwap-shaped bench vs plain Spartan — 2026-06-11

New shape-matched pair (`imod_spartan_modp/.../msshape_c2^12_v2^13` and
`spartan_synthetic/.../msshape`): 2730 random multiplication gates
`a·b ≡ c (mod q)` with `q` = the T256 **scalar-field** modulus and
uniform 256-bit operands, padded to (2^12 cons, 2^13 vars) — the same
shape and padding pattern as MultiSwap k=0 (2715 real rows), with
`log_t_f = 256` → numlimb 8, `t = 2` IntEval iterations. Because the
gate modulus is the native field modulus, plain Spartan proves the
identical statement as one native constraint per gate (`wide` values,
`is_small = false` — claiming small with wide values mis-commits and
fails verification; found the hard way).

**Update (same day):** gate modulus switched from the scalar field `q`
to the T256 **base field** `p` — a foreign modulus the native system
cannot express in one gate, i.e. the representative workload class
(foreign-field / curve-coordinate arithmetic). The plain-Spartan
`msshape` stays native `a·b = c` and reads as a shape-matched
*baseline*, not a same-statement twin. imod cost is modulus-value
independent (confirmed: 434.0 → 428.4 ms, within noise band).

| msshape (criterion median) | imod (k=7) | plain Spartan | ratio |
|---|---|---|---|
| prove  | **428.44 ms** | 16.60 ms | **26×** |
| verify | **37.57 ms**  | 6.58 ms  | **5.7×** |

`IMOD_K` sweep at msshape (w-poly has 13+3=16 total vars, so k=7 forces
`t=2`; k≥8 collapses both opens to `t=1`):

| k | t (w open) | prove | verify |
|---|---|---|---|
| 7  | 2 | 428.4 ms | 37.6 ms |
| 8  | 1 | 397.7 ms | **21.2 ms (−44%)** |
| 9  | 1 | 374.8 ms | — |
| 10 | 1 | **368.5 ms** | — |

Parameters alone bottom out ≈ 368 ms ≈ 22× prove (the LogUp witness
bulk — f_limb's 2^16-entry chunk poly — is k-independent); verify at
k=8 is already 3.2× the native baseline. Reaching <10× prove needs
structural cuts; candidate stack (estimates, unmeasured): merge the w
and q Mod-PCS opens into ONE IntEval instance (stack the two polys,
open at both points via the existing multi-point machinery — halves
chains/GKR/IPA, ~−40%), GKR leaf-layer univariate skip (`bbdbd52`
design, leaf layer dominates the tree), W-build trims. Stack-up lands
roughly at 170-200 ms ≈ 10-12×.

imod one-shot spans (w open 253 ms / q open 189 ms, SNARK 449 ms):
rc_shared 138/106 (logup_gkr 105/86, chunk_commit ~9, reconstr 7/6),
batched_opens 68/54 (w_build 28/22, interleaved_sc 18/15,
merged_ipa 21/16), chain_phase1 10, chain_claims 10, reduction 8.
Five range batches per open at t=2; f_limb chunk poly is 2^16 entries.

Context: the imod row does *integer* modular arithmetic with quotient
advice + limb split + range checks; plain Spartan does one native field
mul. 26×/5.8× is the full machinery overhead on equal shapes — far
better than the 40-100× of the bit-decomp era, and at MultiSwap's real
widths (2048-bit rows ≈ 41k native constraints per row) the per-row
comparison inverts entirely in imod's favor.

Reading: the LogUp-GKR sumcheck is now the single dominant prover cost
(84 ms of 193, ~43%) and is mostly the **fixed per-instance table-side
GKR over the 2^16 multiplicity table** (the witness trees at this
shape are only 12.8k/3.2k entries) — exactly the small-shape overhead
`d8ce6d0` flagged; it amortizes at MultiSwap sizes. Second is the
merged same-column IPA (~39 ms across both opens), a fixed cost per
Mod-PCS open. Candidate levers: share one table-side GKR (or one whole
shared range check) across the w and q opens (halves the 84 ms), and
the GKR leaf-layer univariate skip already designed in `bbdbd52`.
