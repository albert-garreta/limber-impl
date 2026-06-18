# IntMod-Spartan (Phase 2) prover timing breakdown (single-threaded)

Measured breakdown of where `IntModSpartanModpSNARK::prove` spends time, on the
msshape config, **single-threaded**. Single-thread is the reporting basis here
because multi-thread numbers on this machine are not reproducible (see
*Methodology* below); these single-thread figures are stable to a few percent.

> **Headline:** single-threaded, the prover is **MSM/commitment-dominated** —
> the range-check chunk commit (~32%) and the witness commit (~23%) together are
> ~57% of the routine. The LogUp-GKR sumcheck is only ~11%, and the opens ~14%.
> (An earlier *multi-threaded* breakdown reported "GKR ≈ 42%"; that was a
> thermal/scheduling artifact, not the algorithm — see *Multi-thread caveat*.)

## Config & method

- **msshape c12**: `num_cons = 2¹²`, `num_vars = 2¹³`; params `k = 7`,
  `log_t = 32`, `log_t_f = 256` → `numlimb = 8`, derived `s = 7`, `log_p ≈ 30`
  (the fixed-`k` `setup_msshape` default, not `derive_optimized`).
- **Single-threaded** (`RAYON_NUM_THREADS=1`), Apple Silicon, `target-cpu=native`.
- Current code: HEAD includes `cd3b093` (the w/q opens are batched into one
  shared range check + one combined IPA).
- Spans from the `RUST_LOG=info` one-shot block in `benches/imod_spartan_modp.rs`
  (one setup/prove/verify, no criterion iteration). **Single (un-warmed) runs —
  treat absolute ms as ±10–20%; the ranking and proportions are the robust part.**

## Breakdown (msshape c12, single-thread, ≈645 ms full routine)

The timed routine is `msshape_witness` (gen) → `IntModR1CSWitnessModp::new`
(commit `w`,`q`) → `prove`. The `prove` span is ≈490 ms; the witness commit
(~150 ms) and gen (~few ms) happen before it. Full flat breakdown — every span
its own line, sorted by cost:

| # | component | span | ms | % |
|---|---|---|---:|---:|
| 1 | range-check chunk commit | `rc_chunk_commit` | 195 | 30% |
| 2 | witness commit (`comm_w`+`comm_q`) | `imod_modp_wq_commit` | ~150 | 23% |
| 3 | open sumcheck / W-build | `imod_pcs_batched_opens` − IPA | 82 | 13% |
| 4 | LogUp-GKR range check | `rc_logup_gkr` | 76 | 12% |
| 5 | a/b value commit | `imod_pcs_ab_commit` | 33 | 5% |
| 6 | reduction (limb-split→DynPrime) | `imod_pcs_reduction` | 27 | 4% |
| 7 | chain build (partial-eval) | `imod_pcs_chain_build` | 27 | 4% |
| 8 | mult-table commit | `rc_mult_commit` | 14 | 2% |
| 9 | IPA opens | `hyrax_prove_ipa` (×6) | 11 | 2% |
| 10 | range-check recon/bit-validity | `rc_shared` residual | ~9 | 1.4% |
| 11 | witness gen | (unspanned) | ~5 | 0.8% |
| 12 | SNARK outer sumcheck | `imod_modp_outer_sumcheck` | 3 | 0.5% |
| 13 | prime sampling + chain claims | residual | ~3 | 0.5% |
| 14 | SNARK inner sumcheck | `imod_modp_inner_sumcheck` | 2 | 0.3% |

Sum ≈ 638 ms ≈ the 645 ms routine (rounding + un-spanned bits).

- **MSM commits** (rows 1,2,5,8 = chunk 195 + witness 150 + a/b 33 + mult 14 =
  **392 ms, ~61%**) dominate single-threaded — consistent with the flamegraph
  (~45% of single-thread self-time is T256 curve adds).
- **Two of those commits are duplicates** (every range-checked poly is committed
  once in value/limb form and again in chunk form): the witness commit (~150 ms)
  duplicates the `f_limb` chunks inside `rc_chunk_commit`, and the a/b value
  commit (`ab_comm`, 33 ms) duplicates the `Ab` chunks. The commit-sharing levers
  below target these.
- **Witness gen** is *timed* (the gen+commit bench fix) but **un-spanned**, so
  it's the ~few-ms residual. (For `multiswap` gen is large — the real RSA-2048
  exponentiation — but that's a different bench.)
- Single un-warmed run → ±10–20% on absolutes (chunk commit was 210 in an earlier
  run, 195 here).

## What `cd3b093` (batched w/q opens) changed

Controlled same-machine A/B (single-thread, msshape c12), batched vs the prior
two-independent-opens:

| | separate (`f045217`) | batched (`cd3b093`) | Δ |
|---|---:|---:|---|
| LogUp-GKR | 93 (two: 56+37) | 75 (one shared) | −18 |
| combined IPA opens | 127 (two: 71+56) | 91 (one combined) | −36 |
| prove span total | 571 | ~507 | **−64 (~11%)** |

It pays the fixed per-open cost (the 2¹⁶-table GKR + the same-column IPA) once
instead of twice. The witness commit is untouched (still two commits).

## Optimization levers (single-thread view)

The commits dominate single-thread, so commit-side levers matter most here:
- **Witness/`f_limb` commit sharing (`log_t = 16`).** At `log_t = CHUNK_BITS`,
  the witness limbs *are* the range-check chunks, so `comm_w`/`comm_q` and the
  `f_limb` chunk commit are the same polynomial → commit once. Attacks the
  ~151 ms witness commit + the duplicate inside the ~210 ms chunk commit. (See
  the Phase-3 follow-up in `imod_followups.md`.)
- **Batched range-check stacking ("Thread A")** — collapse the per-poly/per-layer
  chunk commits further.
- **GKR (75) and opens (91)** are secondary single-threaded; not the place to
  push first.

## Multi-thread caveat (why single-thread)

Multi-threaded wall-clock on this machine is dominated by measurement confounds,
not the algorithm:
- **Thermal throttling across the config sweep** — msshape runs last, after
  prior configs pin all 14 cores; it measures throttled. This is what inflated
  the multi-thread GKR fraction to the bogus "~42%".
- **Rayon idle-worker spin** — even serial code runs slower under a 14-thread
  pool (idle workers spin-wait, contend for memory bandwidth).
- **Size-dependent GKR scaling** — GKR is 2× *faster* parallel at `N=2²⁰` but
  +48% *slower* at `N=2¹⁶` (bandwidth-bound loops; crossover ~2¹⁸). msshape's GKR
  lives at ~2¹⁶–2¹⁷. `PAR_THRESHOLD = 2¹²` is the right global compromise —
  raising it killed the `N=2²⁰` win.

Net: the prover gets only ~1.24× from 14 cores. Always use a **controlled
same-machine A/B** for before/after — criterion's `change` vs a stale baseline
misleads. See the `feedback-benchmark-single-threaded` memory.

## Reproduce

```
RAYON_NUM_THREADS=1 RUST_LOG=info \
  cargo bench --bench imod_spartan_modp -- zzz   # one-shot spans, no criterion
```
The msshape config is the last block; `imod_spartan_modp_prove` is the prove
span, `imod_modp_wq_commit` the witness commit (outside it).
