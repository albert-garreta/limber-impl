# Zinc+ comparison — measured numbers, mechanism, and open items

Concurrent work: **Zinc+** ("SNARKs for Polynomial Rings", NethermindEth,
eprint 2026/855). An integer/ring SNARK (code-based IPRS commitment + AIR/UCS).
This records a head-to-head against our integer Mod-R1CS (`IntModSpartanModpSNARK`,
Hyrax Mod-PCS), all **single-threaded** (`RAYON_NUM_THREADS=1`,
`target-cpu=native`), measured on the same Mac.

## How the Zinc+ numbers were obtained (reproducibility)

- Repo `github.com/NethermindEth/zinc-plus`. **`main` HEAD does not compile** —
  it pins the `crypto-primitives` git dep to a June rev whose `PrimeField` API
  (no `Modulus`, `cfg()` moved to a trait) is incompatible with the zinc-plus
  code at every commit tried.
- **Fix used:** check out `7eadc16` (2026-05-18, the release the paper figures
  came from) and pin `crypto-primitives` in the root `Cargo.toml` to
  `rev = 2cf39db886a76dc3e961cbb9c86fb5ab042381ef` (2026-03-03, #31 — the rev
  that was `main` HEAD continuously March→June, i.e. what the May code was
  written against; it has `type Modulus` and `cfg()`/`modulus()` on `PrimeField`).
- Build/run: `RAYON_NUM_THREADS=1 RUSTFLAGS="-C target-cpu=native" cargo bench
  --bench e2e --features "parallel simd unchecked iprs-rate-1-8"`.
- This was a throwaway clone (not committed anywhere in our repo).

## Measured single-thread numbers

### Zinc+ (their UAIRs)

| workload | size | prove | verify | proof (zstd) |
|---|---|---|---|---|
| ECDSA MSM only (`RealEcdsa`, native 𝔽_p) | 2⁹ | **24.5 ms** | 1.78 ms | 67 KB |
| SHA-256 only (`RealSha256`) | 2⁹ | 97.9 ms | 6.25 ms | 326 KB |
| SHA+ECDSA (`ShaEcdsa`) | 2⁹ | 143.6 ms | 7.08 ms | 371 KB |
| MulModN 2048-bit (4096 all-real) | 2¹² | 478 ms | 190 ms | 1.17 MB |
| **MulModN 2048-bit (4228 real + pad)** | 2¹³ | **1.10 s** | 483 ms | 1.21 MB |
| MulModN 256-bit | 2¹² | 49.5 ms | — | — |

`MulModN` is a UAIR we wrote (one `a·b ≡ c (mod N)` per row, `assert_zero(a·b −
c − k·N)`) to probe big-composite-modulus cost, since Zinc+ ships no big-int
benchmark. Padding to 2¹³ with only 4228 real rows costs ~the same as 8192
all-real (1.0956 s vs ~1.07 s): **their prover cost is padded-size-dominated**
(IPRS encodes all 8192 values/column; the sumcheck folds the full MLE).

### Ours (`IntModSpartanModpSNARK`, Hyrax Mod-PCS, `main`)

| workload | size | prove | verify | proof |
|---|---|---|---|---|
| ECDSA MSM (256-bit, 4599 rows) | 2¹³ | 559 ms | 44 ms | ~KB |
| Multiswap k=0 (2048-bit, ~5028 real rows) | 2¹³ | 4.35 s¹ | 71 ms | ~KB |

¹ full pipeline = witness-gen (4.7 ms) + witness commit (813 ms) + prove-proper
(~3.6 ms span). vs paper 𝔽_p emulation ≈ **10,152,845 constraints** → 8192 imod
rows ≈ **~1240× fewer constraints** (the integer-collapse win, shared with any
integer SNARK).

## Head-to-head verdicts

### ECDSA — they win ~23× (their *specialization*, not pure engineering)
Their ECDSA sets the projecting prime **q = p** (the secp256k1 base prime, via a
`fixed_prime` module) — so the EC group law is an exact 𝔽_p identity, **no
quotients, no range checks** (their `summary.tex`: "8 witness columns, all
𝔽_p-typed, no range checks"). A field-agnostic commitment is what *lets* them
pick q = p; our Hyrax pins us to T256's scalar field, so our EC arithmetic must
go through the integer/random-prime path with range checks. The 23× is mostly
this specialization — denied for composite moduli.

### Multiswap (big-int, multi-modulus) — Pareto, ~4× prove
Matched at 2¹³ (same real-row count + power-of-2 padding), single-thread:

| | prove | verify | proof |
|---|---|---|---|
| Ours | 4.35 s (3.6 s proper) | **71 ms** | **~KB** |
| Zinc+ | **1.10 s** | 483 ms | 1.21 MB |
| ratio | ~4× them | ~7× us | ~1000× us |

Composite N forbids their q = N native trick, so both run the general integer
mode → the gap drops from 23× (ECDSA) to ~4× — the pure engineering gap
(their code-PCS + cheap IPRS bound + SIMD vs our Hyrax + LogUp-GKR). **We win
verify (~7×) and proof size (~1000×).** A legitimate complementary design point:
Zinc+ = fast prover / big proof / slow verify; ours = small proof / fast verify
/ slower prover.

## Where the time goes (single-thread span breakdowns)

### Their prover is sumcheck-bound; PCS is only ~26%
ECDSA step breakdown: commit 4.6 ms (18%), **sumcheck 15.65 ms (60%)**, opens
~2 ms (8%), ring/projection/proximity ~3.7 ms (14%). The code-PCS is *cheap in
time* (the proof-size cost is the tradeoff). They have no explicit range check —
the value bound rides on the IPRS commitment.

### Our prover is range-check + commit bound (multiswap 2¹³, 2048-bit)
| component | time | share of 4.35 s |
|---|---|---|
| **range check `rc_*`** | **~2.1 s** | ~48% |
|   – `rc_chunk_commit` (Hyrax MSM of chunks) | 1.33 s | |
|   – `rc_logup_gkr` | 655 ms | |
|   – `rc_reconstr` / `rc_mult_commit` | ~110 ms | |
| PCS opens (rest of `wq_open`) | ~1.48 s | ~34% |
| witness commit (`wq_commit`) | ~855 ms | ~20% |
| outer + inner sumcheck + spmv | ~18 ms | ~0.4% |

### Our verifier is PCS-bound; the direct A/B/C eval is negligible
| span | time | share of ~70 ms |
|---|---|---|
| `imod_modp_wq_verify` (Mod-PCS open check) | ~55–63 ms | ~88% |
| `imod_modp_eval_matrices` (verifier computes A/B/C) | **1–2 ms** | ~2–3% |
| sumcheck verify | ~few ms | rest |

The non-preprocessing "SPARK-equivalent" (verifier evaluating sparse A/B/C
directly) is only ~1–2 ms here — immaterial at this scale (O(nonzeros), ~tens of
thousands of entries). Would grow at much larger circuits (the argument for a
preprocessing/SPARK version) but does not affect the verify-time win.

## Mechanism findings

- **Integer commitment vs range checks.** Zinc+ commits *over ℤ* (`IprsCode` =
  pseudo-RS over the integers; `Int<W>` cells), so the value bound rides on the
  commitment: the prover reveals a random linear combination and the verifier
  checks its magnitude (bit-width). No explicit per-value range proof. We commit
  *field elements* (Hyrax/Pedersen), so we must prove the bound with an explicit
  LogUp-GKR range check (~48% of our prover).
- **No rationals hole.** Because they commit integers, a "rational" like ½ can't
  be committed (it's not an integer); the magnitude/proximity check rules out
  oversized values. Subtlety: the bound is *loose* (bit-width + slack), enough
  for mod-q no-wraparound but not tight ranges — so SHA bits still need explicit
  booleanity (why SHA is ~4× their MSM). And soundness needs a *random* prime; the
  `fixed_prime` branch we measured on is cost-representative of a sound random-
  256-bit-prime run (same field size), and a single random 256-bit prime is
  sound for bounded ~4096-bit differences (Pr[divides] ≈ 2⁻²⁴⁸).
- **Big composite moduli are clean for them.** Expressing 2048-bit `mod N` was
  just "widen the `Int<W>` params" — no wall. Cost scales ~linearly in operand
  bit-width. The hoped-for "their framework can't do big composite moduli"
  advantage does **not** materialize.

## Open items / performance recovery

- **`main` is ~2× slower on the imod prover** than the collaborator's out-of-repo
  `e9e9be5` (the version the paper figures came from; an unpushed orphan).
  prove-proper on `main` ≈ 3.6 s; `e9e9be5` ≈ ~1.8 s (→ multiswap ~2 s total,
  the "~2 s" we remembered). The regression is in prove-proper (gen/commit ruled
  out — see [imod_perf_log.md](imod_perf_log.md) / the figure-repro-gap note).
  **Recovering it puts multiswap prove at ~2 s → ~1.8× vs Zinc+'s 1.10 s (near
  prove-parity)**, with verify/proof wins intact. Needs the collaborator to push
  `e9e9be5` for a span diff.
- **The lever is the commit MSMs — the range check stays.** `rc_*` is ~48% of
  prove, but it splits into a *commit* part (`rc_chunk_commit` Pedersen MSM,
  1.33 s) and the *GKR* part (`rc_logup_gkr`, 655 ms). **Mod-PCS-over-Brakedown**
  replaces the Pedersen-MSM commits (`wq_commit` ~855 ms + `rc_chunk_commit`
  ~1.33 s) with hash-based commits → projects prove toward ~2.5–2.9 s (~1.5×).
  The **range check is NOT dropped** — it is load-bearing for soundness (proves
  committed integers are bounded; without it the mod-random-prime reduction is
  unsound). Zinc+'s code-commitment *norm bound* is a different bound *mechanism*,
  not "no range check", and is not something we get by swapping commitments.
  Brakedown only speeds the range check's *commitment* step. Same rewrite tracked
  in [project-brakedown-pcs] / `brakedown_design.md`.
- **Strategic framing for the paper:** point the "we win" comparison at
  native-emulation systems (arkworks/xJsnark — the ~10M→8k constraint collapse).
  Versus Zinc+, frame as concurrent complementary design points (their fast
  prover vs our small proof + fast verify), near prove-parity with the recovered
  baseline.

## Update (2026-07-13): (k, T) retune shrinks our side

A measured (size × k × log_t) sweep found `log_t = 64, k = 9` optimal at every
size and bit-width (see `imod_followups.md`, "IntEvalParams (k, T) retune").
New single-thread numbers with the re-tuned defaults:

| | prove | verify | proof |
|---|---|---|---|
| Ours (multiswap 2¹³, re-tuned) | **3.11 s** | **42.5 ms** | ~KB |
| Zinc+ (MulModN 2¹³) | 1.10 s | 483 ms | 1.21 MB |
| ratio | ~2.8× them | **~11× us** | ~1000× us |

ECDSA MSM: 386 ms / 24 ms (was 545/43) — Zinc+ gap ~16× (was 23×). No protocol
change; parameters only. The T=2^16 witness-commit-reuse idea was implemented,
measured, and reverted (dominated by T=2^64 — details in imod_followups.md).

## Update (2026-07-22): committed-chunk representation

The witness commitment now IS the range check's 16-bit chunk commitment
(the duplicate 64-bit limb MSM is gone; see imod_followups.md
"Committed-chunk representation"). New single-thread numbers, same
(k=9, T=2^64) — re-confirmed optimal post-change:

| | prove | verify | proof |
|---|---|---|---|
| Ours (multiswap 2¹³, chunked commit) | **1.37 s** | **39.4 ms** | ~KB |
| Zinc+ (MulModN 2¹³) | 1.10 s | 483 ms | 1.21 MB |
| ratio | ~1.25× them | **~12× us** | ~1000× us |

ECDSA MSM: 189 ms / 21.8 ms — Zinc+ gap ~8× (their native-q=p
specialization). Cumulative since the 4.35 s starting point: −69% prove
with verify/proof-size wins intact. (1.37 s reflects seven rounds:
chunked witness commit 3.11→2.43 s, the limb-split/scalar-cast rewrite
→2.13 s, the a/b chunk-only layer commitments →2.03 s, fixed-width I256
chain arithmetic →1.87 s, the MSM-window/chunk-build/scalar-table
micro-fixes →1.71 s, vartime bucket adds in the small-scalar MSM
→1.47 s, and the lockstep LogUp-GKR + leaf-table fix →1.37 s — see imod_followups.md "Committed-chunk
representation" and its follow-ons.)

Current prover breakdown (multiswap 2¹³, single-thread spans; supersedes
the 4.35 s table above for the current code):

| component | time | share of 1.37 s | justification (per-op model) |
|---|---|---|---|
| `wq_commit` (16-bit chunk MSM, the only witness commit) | 377 ms | ~28% | 2·2²⁰ points × ~0.9 actual adds/point (witness sparsity) at ~165 ns vartime mixed add; next ~1.5× needs batch-affine adds |
| `rc_logup_gkr` (lockstep multi-tree) | 460 ms | ~34% | ~2.4M fraction-tree leaves × ~4–5 field mults at ~40 ns — field-mult floor; lever is batched GKR, not constants |
| combined batch open | 304 ms | ~18% | weight build ≈ 2 eq-passes/target (~6M fused mul-adds) + sumcheck ≈ 4 mults/element over 2.45M elements — matches mult floor |
| `ab_commit` (per-layer `a`/`b` chunk MSMs) | 100 ms | ~7% | ~300k-point MSMs + chunk builds, same MSM constant |
| reduction (limb split + casts + integer eval + sumcheck) | ~91 ms | ~5% | format conversions at ~1 Montgomery mul/element over 2¹⁹ limbs ×3 domains; `int_v'` must stay `BigInt` (multi-thousand-bit) |
| chain build (I256 integer partial-evals) | 44 ms | ~3% | 8.4M fixed-width mult-adds ≈ 5 ns each — arithmetic floor |
| GKR witness prep + mult commit + Spartan SCs | ~40 ms | ~2% | table-lookup chunk prep (4 ms) + one 2¹⁶-point MSM + 2¹³-size sumchecks |

Verify 42 ms: batch-open verification ~28 ms (~67%), range-check verify
~4 ms, matrix eval 1–2 ms. Total Pedersen MSM work is now ~500 ms
(~34%) — the honest size of the Mod-PCS-over-Brakedown lever (projects
prove toward ~1.0 s, i.e. Zinc+ parity); batch-affine MSM internals
(~110–120 ms more) is the in-place alternative. The GKR is now lockstep-batched:
multithreaded it scales 3.2× (143 ms), and multithreaded prove-proper
is ~530 ms total.
