# Arkworks/Garuda baseline — measured (2026-08-13)

Supports the paper's Arkworks row in tab:multiswap-bench. Harness:
`bbuenz/rsa-exp-snark` (GR1CS RsaExpCircuit via r1cs-std emulated-fp,
Garuda/Pari from `alireza-shirzad/garuda-pari`, BLS12-381), run locally
with a small patch printing `proof.serialized_size()` (compressed +
uncompressed) after verify. Throwaway clone under the job tmp dir, kept
out of this repo, sibling stack per its setup.sh (isolated from the
zinc-plus clone's crypto-primitives pin — they need different revs).

## The actual MultiSwap statement in this framework

`352,4` (4 Wesolowski 352-bit exps mod RSA-2048, square+multiply per
bit = 2,816 modmuls; H/Hp are field-native at 255 bits, <1% of count):

- interp mul (128x16-bit limbs, dense rows): **15,695,748 constraints**
- schoolbook mul (26x79-bit limbs, sparse rows): **25,208,818
  constraints** — matches the paper row's "25 million" exactly; the
  row is the schoolbook wiring of this statement.

## Garuda proof size: exactly +264 B compressed per log2 level

| constraints | bucket | compressed | uncompressed |
|---|---|---|---|
| 97,193 | 2^17 | 5,056 B | 6,976 B |
| 186,225 | 2^18 | 5,320 B | 7,336 B |
| 364,289 | 2^19 | 5,584 B | 7,696 B |
| 720,417 | 2^20 | 5,848 B | 8,056 B |
| 1,438,658 | 2^21 | 6,112 B | 8,416 B |
| 2,863,170 | 2^22 | 6,376 B | 8,776 B |

Formula: compressed = 5056 + 264·(ν−17), uncompressed = 6976 +
360·(ν−17), ν = ceil(log2 n). At the paper row (25.2M, ν=25):
**7,168 B compressed / 9,856 B raw (~7 KB)**.

## Full-size run: RESOLVED (2026-08-13, same Mac, RAYON_NUM_THREADS=1)

`352,4` schoolbook completed locally (peak RSS 13.5 GB): keygen 904 s,
**prove 231.35 s, verify 24.81 ms, proof 7,168 B compressed / 9,856 B
raw** — proof matches the ladder formula exactly, and prove confirms
the paper's original 230 s as a single-threaded schoolbook Garuda run.
Schoolbook is the best wiring for Garuda by far (same-statement A/B:
prove 7-8x faster than interp despite 1.6x more constraints — sparse
rows beat dense 255-term interp rows). Paper row set to
25.2M / 231s / 25ms / 7.2KB. NOTE: verify one-shot estimates from
smaller sizes under-predicted (12-13 ms est vs 24.8 ms measured) —
quote measurements, not extrapolations.

## Memory: interp keygen is ~9x per constraint (measured 2026-08-14)

Peak memory footprint (`/usr/bin/time -l`, single-thread, same Mac):
schoolbook 4.7-4.8 KB/constraint (0.72/1.42/2.74 GB at 150k/294k/580k);
interp ~45 KB/constraint (4.26 GB at 97k, 16.3 GB at 364k) — ~9x per
constraint, ~6x absolute on the same statement despite 1.6x fewer
constraints. Cause: keygen materializes per-NONZERO data and interp
rows are 255-term dense. Full-size interp (15.7M) extrapolates to
~700 GB — NOT runnable on 24 GB (an unguarded attempt swap-froze the
machine); full-size schoolbook measured 65.4 GB peak footprint /
13.5 GB RSS (survives via the memory compressor, machine sluggish).
Schoolbook full-size fit over-predicts ~2x (120 GB vs 65 GB measured)
— conservative, acceptable for a safety gate. Interp full-size numbers
are therefore extrapolations only: prove ~590-620 s single-thread
(37 us/constraint measured at 364k-720k), proof 6,904 B (one bucket
lower), verify unknown (its per-level slope measured STEEPER than
schoolbook's: 10.0->13.6 ms across one doubling). Best wiring for the
paper row stays schoolbook on every axis that matters.

## Row-consistency problem (superseded by the full-size run above)

Measured Garuda verify: 8.0 ms (97k) → 16.2 ms (2.86M), growing with
log n — **never near the row's 2 ms at any size**. 2 ms + 0.2 KB are
Groth16 numbers (192 B, O(1) pairings). Either the 230 s / 2 ms pair
was measured with Groth16 (then drop "(with Garuda)" and use 0.2 KB),
or the row is Garuda (then verify ≈ 20 ms, proof ≈ 7 KB, and 230 s
needs re-confirmation from the original run). Prover time is NOT
locally checkable: this 24 GB machine swaps above ~1.5M constraints
(prove 22.8 s at 720k → 193.7 s at 1.4M, memory-degraded); the 5.7M
keygen died. Garuda prove measured multithreaded here: 10.1 s @ 364k,
516 s @ 2.86M (memory-pressured; don't extrapolate).
