# spartan-inteval: SNARKs for Integers on Spartan

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

- **NeutronNova zkSNARK** — non-recursive
  [NeutronNova](https://eprint.iacr.org/2024/1606) folding for uniform
  computations: many instances of one step circuit are multi-folded
  into a single R1CS instance proved with Spartan, amortizing the
  prover across the batch.

- **Precomputable / online witness split** — both protocols expose
  `setup` → `prep_prove` → `prove`, so witness material known ahead of
  time is synthesized and committed once and reused across proofs.
  This is the pattern [Vega](https://eprint.iacr.org/2025/2094) relies
  on for low-latency proving.

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
| `sha256_neutronnova` | NeutronNova over 32 SHA-256 step circuits |

Override thread counts with `BENCH_THREADS` (comma-separated):

```bash
BENCH_THREADS=1,8 RUSTFLAGS="-C target-cpu=native" cargo bench --bench imod_spartan
```

## References

*SNARKs for Integers* — the protocol this repository implements.

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
