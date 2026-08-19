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
| Multiswap k=0 (2048-bit, 4831 real rows) | 2¹³ | 4.35 s¹ | 71 ms | ~KB |

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

## Update (2026-08-03): the Brakedown instantiation — measured, and a negative result worth having

The full Mod-PCS now also runs over Brakedown (hash commitments, no
elliptic-curve operations, non-hiding) behind a commitment-backend
interface; same protocol, same committed-chunk layout, per-target
tensor-IOPP final openings. Measured on the same multiswap 2¹³
workload, single-threaded (BDPCS=1 on the multiswap bench):

| instantiation | prove (total) | verify | proof |
|---|---|---|---|
| Ours / Hyrax (Pedersen) | **1.34 s** | **37.9 ms** | **~KB** |
| Ours / Brakedown v1.1 | 2.11 s | 182 ms | 21.75 MB |
| Zinc+ (IPRS) | 1.13 s | 482 ms | 1.21 MB zstd |

**The projection that hash commitments would reach Zinc+ prover parity
was wrong, and the reason is the finding:**

1. After the small-scalar optimization campaign, our Pedersen chunk
   commit (16-bit scalars, vartime buckets) costs ~380 ms — CHEAPER
   than Brakedown's encode+hash (~1.1 s) at the same data. Group
   arithmetic on small scalars beats hashing when the hashing is done
   at field width.
2. The structural cost is the **field-width tax**: our chunk values are
   16-bit, but field-level Brakedown encodes and Keccak-hashes them as
   256-bit field elements — ~16× more bytes than the information
   content. Zinc+'s IPRS is **integer-native** (small `Int<W>` cells,
   SIMD): that, not "hash vs group", is their prover advantage on this
   axis.

v1.1 applied the cheap fixes: opening-data caching across
commit/prove (−0.86 s), one-time code-matrix sampling moved to setup
(−0.21 s, 208 ms measured), BLAKE3 column/tree hashing with Keccak kept
for Fiat–Shamir (−0.09 s). The small BLAKE3 delta confirms the commit
is ENCODE-dominated — the expander's field multiplications pay the same
width tax as the hashing. Remaining engineering levers: merging
same-transcript-moment commitments (9 trees → 4, ≈ halves the proof)
and 16-bit serialization of systematic columns (~3–4 MB more); neither
touches the ~2 s prover floor at field width. Numbers are end-to-end
verified (roundtrip + tamper tests through the full SNARK driver).

**Paper framing:** one protocol, measured at both ends of the
commitment design space, with an explanation of why the frontier sits
where it does — an integer-native code commitment (their design) or a
small-scalar-optimized group commitment (ours) both beat a field-level
hash commitment on small-valued data.

## Update (2026-07-22): committed-chunk representation

The witness commitment now IS the range check's 16-bit chunk commitment
(the duplicate 64-bit limb MSM is gone; see imod_followups.md
"Committed-chunk representation"). New single-thread numbers, same
(k=9, T=2^64) — re-confirmed optimal post-change:

| | prove | verify | proof |
|---|---|---|---|
| Ours (multiswap 2¹³, chunked commit) | **1.34 s** (1.46 throttled) | **37.9 ms** | ~KB |
| Zinc+ (MulModN 2¹³, re-measured) | **1.13 s** (1.20 throttled) | 482 ms | 1.21 MB zstd (4.76 MB raw) |
| ratio | **~1.19–1.22× them** | **~13× us** | ~1000× us |

Official cold-machine pair (2026-07-24, rested machine, back-to-back):
ours 1.344 s / 37.9 ms; Zinc+ 1.131 s / 482 ms.

**Accounting notes (2026-08-03/06).** (1) ECDSA: the
`ecdsa_msm_prove_time` test originally excluded the witness commitment
from its timer; re-measured commit-inclusive (single-thread, k=9):
commit 59 ms + prove 187 ms = **246 ms**, verify 20.6 ms -> gap vs
Zinc+'s q=p specialization (24.5 ms) is **~10×** (earlier "~8×" was
prove-only). (2) MultiSwap: a 2026-08-03 "correction" claiming the
prove ratio was 1.49× DOUBLE-COUNTED the commit and is retracted: the
criterion `prove/` timed region has always been the full pipeline —
witness generation + witness commitment + prove (see the bench
comment) — so the published commit-inclusive numbers and the ~1.19×
cold-pair ratio were correct as originally stated. Zinc+'s e2e prove
also commits inside its timed region (their PCS is ~26% of prove) but
receives a prebuilt trace, so our number including witness generation
is slightly conservative against us. Post zero-block-drop
(2026-08-04): ours 1.219 s full-pipeline vs Zinc+ 1.13 s ->
**~1.08×**.

**Same-hour pairing (2026-07-23):** both sides re-measured back-to-back
on the same machine state after the full optimization series: ours
1.46 s / 41.6 ms, Zinc+ 1.20 s / 510 ms. Both run ~+8% slower on a
thermally loaded machine (sustained benchmarking lowers single-core
boost clocks) versus their rested-machine numbers (1.36 s / 1.10 s) —
two independent codebases drifting by the same factor at the same time
is the signature of DVFS, not code. The RATIO is the stable quantity — ~1.22–1.25× prove under either
condition. ECDSA MSM: 189 ms / 21.8 ms — Zinc+ gap ~8× (their
native-q=p specialization). Cumulative since the 4.35 s starting
point: −69% prove with verify/proof-size wins intact. (1.37 s reflects seven rounds:
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
| `rc_logup_gkr` (lockstep multi-tree) | 427 ms | ~31% | ~2.4M fraction-tree leaves × ~4–5 field mults at ~40 ns — field-mult floor; lever is batched GKR, not constants |
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
multithreaded it scales 3.4× (126 ms), and multithreaded prove-proper
is ~520 ms total.

## Proof-size correction and measurement (2026-08-06)

The "~KB" / "~1000x" proof-size claim was never measured and is wrong;
the bench now measures it (`PSIZE=1`, `eval_arg_component_sizes`).
MultiSwap 2^13, Hyrax path, single proof:

| RC_BLOCK_LOG | proof (raw bincode) | range_check share | commit+prove (one-shot) |
|---|---|---|---|
| 16 (current) | **637 KB** | 551 KB | ~1.26-1.35 s |
| 18 | 351 KB | 266 KB | ~1.28 s |
| 20 (~unblocked) | 273 KB | 187 KB | ~1.39 s |

plus ~1.2 KB of (not yet serializable) sumcheck rounds/evals. The
multi-tree LogUp-GKR dominates (~17 KB round polynomials per tree;
zero-block splitting multiplies tree count -- a prove-time/proof-size
trade nobody measured until now). Corrected comparison: Zinc+ 4.76 MB
raw / 1.21 MB zstd vs ours 0.27-0.64 MB raw -> **we are ~2-17x
smaller, not ~1000x**. Brakedown path: 20.08 MB (k=11). Groth16-class
baselines (original MultiSwap paper via bellman/BLS12-381; arkworks
emulation) are constant ~128-192 B -- three orders below all
code/vector-commitment systems including ours and Zinc+'s.
RC_BLOCK_LOG=18 looks Pareto (half the proof, prove within noise);
adopting it needs a criterion A/B. Real fix: RLC-batch the lockstep
trees' round polynomials into one per round (see gkr_uniskip_plan.md)
-> ~100 KB at full prove speed.

## Faithful-cost configuration (2026-08-07)

The multiswap bench now charges `Hp` at faithful cost and structure
(previously 600 modeled rows): 600 Pocklington-exponentiation rows +
3 chained Poseidon-cost permutations (243 mul rows each -- 81 x^5
S-boxes x 3 muls; MDS/round-constant layers fold into the LCs free;
synthetic operands, real chain shape and modulus p_hash) + 640
mod-0 decomposition bit rows + 10 reconstruction rows mod l. New
instance: **6,210 real rows** (was 4,831), cons still 2^13; columns
7,437 (chained hash rows cost ~1 column/row), vars still 2^13 -- the
variable-padding boundary is NOT crossed.

Zinc+ needs no circuit change: their prover cost is padded-trace
(8,192 rows x 2048-bit width), which upper-bounds the same faithful
workload (their hash rows would be narrower-modulus rows in the same
trace), so their measured 1.13 s stands as the fair number.

Measured (single-thread, criterion full pipeline = witness gen +
commit + prove): **prove 1.347 s, verify 37.8 ms, proof 725 KB**
(range check 639 KB -- more nonzero rows -> fewer dropped zero
blocks). Fair-configuration ratio vs Zinc+: **~1.19x prove**, verify
~13x us, proof ~1.9-6.6x us (raw 4.76 MB / zstd 1.21 MB vs 725 KB).
The previous 4,831-row numbers (1.219 s / 637 KB) remain valid as the
"modmul-core" configuration. At k>0 the H-delta model is still 8
rows/invocation (faithful ~500-650) -- do not quote k>0 without
fixing that.

## Fully wired configuration + fairness re-evaluation (2026-08-07)

The bench circuit is now FULLY wired at k = 0: group-mult operands are
the exponentiation output columns; `Hp` is 4 real square-and-multiply
chains (50-bit exponents, Mersenne moduli 2^61/89/107/127 − 1, with
bit decomposition and reconstruction, via the same `build_exp_circuit`
as the Wesolowski chains); the Poseidon chain is seeded by reducing
exp output 0 mod p_hash; the 639 decomposition bit rows fully decompose
the Poseidon output and all four chain outputs with exact mod-0
reconstruction rows; the final mod-l row reduces the Poseidon output.
Witness: **6,209 rows / 6,204 real columns (~1 column per row — the
witness floor: one fresh value per multiplication output)**; shape
still 2^13 x 2^13. Only remaining unfaithful piece: the k>0 H-delta
model (do not quote k>0).

Prover-side zero exploitation extended to the batched-open bind/eval
passes (elementwise both-zero skips; chunk layouts are ~50% strided
interior zeros from values narrower than their limb budget).

Measured (single-thread, full pipeline): **prove 1.278 s, verify
38.2 ms, proof 672 KB** -> ratio vs Zinc+ **1.13x**.

**Fairness verdict, sharpened:** Zinc+'s 1.13 s is an UNWIRED,
single-modulus LOWER BOUND, not their fair number. (a) Gate-count
corrections provably don't move it: 4,228-real+pad = 8,192-all-real
measured (1.0956 vs ~1.07 s) — their prover pays padded list price,
zeros are not free for hash-based linear-time commitment, while ours
demonstrably harvests zeros (blocking, MSM skips, bind/eval skips).
(b) What padding does NOT absorb for them: wiring (their MulModN probe
— which we wrote — has no copy constraints or transition structure;
the faithful statement is chained everywhere) and modulus mixing
(their probe bakes in one N). Their own wired UAIRs indicate structure
costs them ~an order of magnitude per row: RealSha256 97.9 ms at 2^9
rows vs ~6 ms extrapolated for raw MulModN-256 at the same size.
A faithful wired multiswap in their framework plausibly lands well
above 1.13 s; building it is their burden, not ours. Summary line for
the paper: our 1.278 s is a fully-wired faithful-cost measurement;
their 1.13 s is an unwired lower bound; the true wired-vs-wired gap is
at most 1.13x and likely at or below parity.

## Measured: chained cost-model UAIR in their framework (2026-08-07)

Built `ModMulChainedUair` in the Zinc+ clone (test-uair/src/
modmul_chained.rs, bench "ModMulChained2048"): the unwired probe plus
every structural element the wired multiswap layout needs, at faithful
cost drivers with synthetic values — per-row witnessed modulus column
(mixed RSA-2048 / 2^255−19 / Mersenne-127 schedule), shift-1 chaining
transition a[r+1]=c[r], bit column with binary constraint, and a
reconstruction stand-in column with a shift-1 transition. 7 Int<33>
columns vs the probe's 4; degenerate satisfying witness (cost is
structure-driven — measured value-insensitivity).

Same-session single-thread at 2^13: probe 1.255 s -> **chained
1.983 s (+58%)**. Scaling by the session's warm factor (their official
cold probe = 1.13 s) gives **~1.79 s cold** for the wired-layout cost
model. Refinements could shave it (bit as a binary_poly column,
modulus as public columns) — call it **~1.6-1.8 s** fair-bracket.

**Fair wired-vs-wired verdict: ours 1.278 s vs theirs ~1.8 s — we are
~1.4x FASTER on prove**, plus ~13x verify and ~7x proof size. The
"they win prove" framing inverts once both sides pay for the same
statement structure: their unwired probe hid a +58% structural cost
their architecture cannot avoid (extra wide committed columns), while
our LC-based wiring is free.

Fidelity asymmetry, stated precisely: OUR circuit is a real, fully
wired circuit evaluating a synthetic instance — every row performs
genuine arithmetic on genuine dataflow — with two cost-faithful gadget
stand-ins (Poseidon internals with identity linear layers; Pocklington
side-condition comparisons). THEIRS is a structural cost model with
degenerate values (5·1 = 5 rows; no real dataflow), legitimate for
cost measurement (structure-driven, verified via their padding/value
insensitivity) but a categorically weaker artifact. Our remaining gap
to fully-faithful is small and shape-neutral: real Poseidon constants
(LC coefficients only — cost-identical by construction), the
Pocklington comparison rows (a few hundred, absorbed by padding), and
witness generation from an actual accumulator run.

## Same-session final pair (2026-08-07, warm machine, back-to-back)

| | ours (wired, real circuit) | Zinc+ probe (unwired) | Zinc+ chained (fair) |
|---|---|---|---|
| prove (full pipeline) | **1.315 s** | 1.276 s | **2.060 s** |
| verify | **39.5 ms** | 523 ms | 514 ms |
| proof, raw | **672 KB** | 4.76 MB | 4.84 MB |
| proof, zstd | ~raw (field data) | 1.24 MB | n/a¹ |

¹ The chained cost-model's zstd figure (79 KB) is an artifact: the
degenerate witness values make the opened IPRS columns trivially
compressible. Quote raw for the chained model; the probe's 1.24 MB is
the realistic compressed figure for real data.

Fair (wired-vs-wired) ratios, same thermal state: **prove ours 1.57×
faster; verify ours 13×; proof ours ~7× smaller (raw), ~2× (their
zstd vs our raw)**. Against their unwired lower-bound probe we are at
prove parity (1.315 vs 1.276) while keeping the verify and proof wins.
## γ-RLC GKR round-poly batching lands (2026-08-13)

The "real fix" flagged in the proof-size section is implemented: the
lockstep multi-tree GKR walk (`gkr_prove_multi`) now sends ONE
γ-power RLC of the active trees' cubic round polynomials per round
(γ squeezed per layer, after the previous layer's finals are absorbed
— standard batched-sumcheck soundness, ≤ (#trees−1)/|F| extra loss
per layer). Per-tree layer finals remain (they seed the next layer's
claims and are pinned by the leaf PCS opens), grouped per layer in a
shared `GkrMultiProof`. Round-poly bytes drop from Σ_t Θ(d_t²) to
Θ(max_d²).

Measured, MultiSwap 2^13 Hyrax path, same-machine A/B (criterion,
back-to-back):

| | before | after | change |
|---|---|---|---|
| proof (raw bincode) | 672.4 KB | **174.7 KB** | **−74%** |
| — range_check component | 586.4 KB | 88.7 KB | −85% |
| prove (criterion median) | 491.0 ms | 472.1 ms | −3.5% |
| verify | 22.9 ms | 20.8 ms | −7.7% |

(Absolute times are MULTITHREADED — this A/B ran without
`RAYON_NUM_THREADS=1`, unlike the single-threaded 2026-08-07 pair
(1.315 s) quoted in the paper table; the ~2.7× gap between the two is
thread count, not code. The A/B is internally consistent — both sides
multithreaded, back-to-back — and proof size is thread- and
machine-independent.) The prove/verify wins are real but small:
thousands of per-tree transcript absorbs and per-tree cubic
evaluations collapse to one per round.

Updated scoreboard vs Zinc+ (proof sizes deterministic, timing pairs
unchanged from 2026-08-07): **ours 174.7 KB raw vs their chained
4.84 MB raw → ~28× smaller** (vs probe 4.76 MB raw: ~27×; vs the
probe's realistic 1.24 MB zstd: ~7×). **Paper quotes the zstd-fair
pairing (user decision 2026-08-14): 175 KB vs 1.2 MB → 7×**, with the
raw 4.8 MB disclosed in the table footnote; their zstd is quoted from
the PROBE's realistic data, never the chained run's 79 KB
degenerate-value artifact. Remaining proof-size structure:
combined_open 74.5 KB (√n Hyrax IPA-free opening) + range-check
finals ~85 KB + per_poly 10.3 KB; next lever if ever needed is
RC_BLOCK_LOG=18 (fewer trees → fewer finals) or trimming the
combined-open transcript.

msshape sweep (γ-RLC code, `PSIZE=1`, full-impl proof incl. sumcheck
side): imod 135.2 / 149.1 / 137.8 KB at c2^10 / 2^12 / 2^14 (range
check now 24-40% of the proof, combined_open dominant) vs plain
Spartan (Hyrax) 67.8 / 68.3 / 69.0 KB — **native proof-size overhead
~2.0-2.2×**, down from ~5-6× pre-batching.
