# Limber: SNARKs for Integers on Spartan

A research prototype of **Integer Mod-R1CS** proving — SNARKs whose
constraints are modular arithmetic over arbitrary, per-constraint
integer moduli — built as a fork of
[Microsoft Spartan2](https://github.com/Microsoft/Spartan2). It
implements the protocol from the accompanying *SNARKs for Integers*
paper (Def. 5.4) on top of Spartan's sum-check pipeline.

## Why integer Mod-R1CS

Standard R1CS forces all arithmetic into one fixed prime field, so a
single multiplication modulo a foreign modulus `N` (say, a 2048-bit RSA
modulus) costs thousands of constraints in limb-decomposed form. The
Integer Mod-R1CS relation works over the integers with a **per-row
modulus** `m_i` and a prover-supplied quotient `q_i`:

```text
A·z ∘ B·z = C·z + m ∘ q        (over Z, with bounded norms)
```

so one row *is* one modular multiplication `LC_A · LC_B ≡ LC_C (mod m_i)`,
regardless of how wide `m_i` is. The sum-check is then run modulo a
random prime `p` sampled by the verifier via Fiat–Shamir.

## The base library

The fork retains Spartan2's proving systems, which the integer work
builds on and benchmarks against:

- **Spartan zkSNARK** — a PCS-generic implementation of
  [Spartan](https://eprint.iacr.org/2019/550), a sum-check-based
  zkSNARK with a linear-time prover. Accepts R1CS circuits written
  with [bellpepper](https://github.com/lurk-lab/bellpepper) and works
  with any multilinear PCS (Hyrax engines over Pallas/Vesta/P-256/T-256
  and BN254 are provided). Zero-knowledge is obtained via Nova's
  folding scheme.

- **Precomputable / online witness split** — both protocols expose
  `setup` → `prep_prove` → `prove`, so witness material known ahead of
  time is synthesized and committed once and reused across proofs.

## Benchmarks

All benchmarks use [Criterion](https://github.com/bheisler/criterion.rs)
and report setup / (prep_)prove / verify times plus proof sizes. Run
with native CPU codegen:

```bash
RUSTFLAGS="-C target-cpu=native" cargo bench --bench <name>
```

| Bench | What it measures |
| --- | --- |
| `imod_spartan` | Phase-1 IntMod-Spartan on synthetic modular multiplications |
| `imod_spartan_modp` | Phase-2 driver (`T256DynPrimeEngine` + integer Mod-PCS) on the same shapes |
| `spartan_synthetic` | Plain-Spartan baseline, shape-matched to the imod benches |
| `multiswap_modp` | MultiSwap (RSA-accumulator swap batches, [OWWB20](https://eprint.iacr.org/2019/1494)) with wired 2048-bit square-and-multiply chains — one imod row per `mod N` multiply |
| `logup_gkr` | LogUp-GKR range proof in isolation |
| `sha256_spartan` | Spartan over SHA-256 (1–2 KiB messages) |

Override thread counts with `BENCH_THREADS` (comma-separated):

```bash
BENCH_THREADS=1,8 RUSTFLAGS="-C target-cpu=native" cargo bench --bench imod_spartan
```

## Reproducing the paper's numbers

All numbers quoted in the paper are **single-threaded**
(`RAYON_NUM_THREADS=1`) with native codegen
(`RUSTFLAGS="-C target-cpu=native"`). Multi-threaded runs are not
comparable across configurations (thermal throttling and rayon
spin-up confound the ratios), so always pin the thread count when
reproducing.

**Native-overhead figure (`fig:nativeoverhead`) and the msshape
plots/table** (`docs/plots/msshape_*`):

```bash
RAYON_NUM_THREADS=1 ./scripts/regen_msshape_plots.sh
```

This runs the shape-matched pair of sweeps
(`cargo bench --bench imod_spartan_modp -- msshape` vs
`cargo bench --bench spartan_synthetic -- msshape`) and renders the
figures via `scripts/plot_msshape.py`; see the script header for
knobs. Expected ballpark (Apple Silicon, 2026-08): 5.9–8.1× prover
overhead over plain Spartan at 2^10–2^14 constraints, verify under
30 ms vs 15–17 ms, proof 135–149 KB vs ~68 KB.

**MultiSwap table (`tab:multiswap-bench`), our rows**:

```bash
RAYON_NUM_THREADS=1 RUSTFLAGS="-C target-cpu=native" cargo bench --bench multiswap_modp
```

Set `PSIZE=1` to print serialized proof sizes and `KSWEEP=1` to sweep
the reduction parameter `k` (see the bench's module docs for what is
wired faithfully vs modeled by operation count). The Arkworks/Garuda
baseline row comes from an external harness.

**Zinc+ comparison**: our side is the `multiswap_modp` run above at
2^13 rows; the Zinc+ side requires pinning specific revisions of
their repo to build at all.

## References

[Limber: Low Overhead SNARKs for Integers from Any PCS](https://eprint.iacr.org/2026/1635) — the protocol this repository implements.

[Spartan: Efficient and general-purpose zkSNARKs without trusted setup](https://eprint.iacr.org/2019/550) \
Srinath Setty \
CRYPTO 2020

[Scaling Verifiable Computation Using Efficient Set Accumulators](https://eprint.iacr.org/2019/1494) \
Alex Ozdemir, Riad S. Wahby, Barry Whitehat, Dan Boneh \
USENIX Security 2020

## License

MIT, inherited from the upstream
[Spartan2](https://github.com/Microsoft/Spartan2) project — see
[LICENSE](LICENSE).
