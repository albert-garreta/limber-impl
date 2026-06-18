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

## Breakdown (msshape c12, single-thread, ≈660 ms full routine)

The timed routine is `msshape_witness` (gen) → `IntModR1CSWitnessModp::new`
(commit `w`,`q`) → `prove`. The `prove` span itself is ≈509 ms; the witness
commit (≈151 ms) and gen (~few ms) happen before it.

| component | ms | % | span | kind |
|---|---:|---:|---|---|
| chunk commit (range check) | 210 | ~32% | `rc_chunk_commit` | **MSM** |
| witness commit (`comm_w`+`comm_q`) | 151 | ~23% | `imod_modp_wq_commit` | **MSM** |
| batched opens (one combined IPA) | 91 | ~14% | `imod_pcs_batched_opens` | open sumchecks + IPA |
| LogUp-GKR (one shared) | 75 | ~11% | `rc_logup_gkr` | sumcheck |
| chain phases (per-prime setup) | 66 | ~10% | `imod_pcs_chain_phase1/claims` | — |
| reduction (limb-split→DynPrime) | 30 | ~5% | `imod_pcs_reduction` | — |
| mult commit | 14 | ~2% | `rc_mult_commit` | **MSM** |
| SNARK outer/inner sumcheck + wit-gen + misc | ~13 | ~2% | `imod_modp_*` / unspanned | — |

- **MSM commits** (chunk 210 + witness 151 + mult 14 = **375 ms, ~57%**) are the
  dominant single-thread cost — consistent with the flamegraph (~45% of
  single-thread self-time is T256 curve adds).
- **Witness gen** is *timed* (the gen+commit bench fix put `msshape_witness`
  inside the measured routine) but has **no span**, so it's the ~few-ms residual,
  not a line item. (For `multiswap`, gen would be large — the real RSA-2048
  exponentiation — but that's a different bench.)

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
