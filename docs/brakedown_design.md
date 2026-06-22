# Brakedown PCS — design & parameters

A standalone, group-free polynomial commitment scheme implementing
`PCSEngineTrait<E>` using only `E::Scalar` (no curve / MSM). Built to validate
the PCS-agnostic design and as a faster commit path for our commit-dominated
prover (commits are ~61% of single-thread prove; Brakedown commits via a linear
code + Merkle hash instead of a Pedersen MSM).

Milestone 1 (this work): standalone PCS, tested against plain Spartan. The
Mod-PCS (IntEval) integration is deferred — it relies on commitment
homomorphism (RLC batched opens, same-column IPA, stacked commitments) that a
Merkle commitment does not provide; that is a separate ~40–60% rewrite of
`integer_modpcs.rs`.

## Linear-time code (GLSTW Spielman/expander)

Parameters ported from the Brakedown authors' reference impl
[`conroi/lcpc`](https://github.com/conroi/lcpc) (`lcpc-brakedown-pc`,
`codespec.rs` + `matgen.rs`), which implements GLSTW21
("Brakedown: Linear-time and field-agnostic SNARKs for R1CS", eprint 2021/1043).

### Construction (systematic, codeword length `⌈R·n⌉`)

```
Enc(x)  with |x| = n:
  if n <= baselen (20):           base code (small dense systematic, rate R)
  else:
    z      = A · x                # precode A: n -> m = ⌈α·n⌉, cn nonzeros/row
    z_enc  = Enc(z)               # recurse, length niprime = ⌈R·m⌉
    v      = B · z_enc            # postcode B: niprime -> miprime, dn nonzeros/row
    return [ x | z_enc | v ]      # length n + niprime + miprime = ⌈R·n⌉
```

The sparse matrices `A`, `B` are sampled deterministically from a public seed
(in the commitment key) so prover and verifier agree. Each matrix row holds a
fixed number of random nonzero field entries at random columns.

### Per-level dimensions (input length `ni`, `ceil_muldiv(a,p,q)=⌈a·p/q⌉`)

```
mi       = ⌈α·ni⌉                       # precode output
niprime  = ⌈R·mi⌉                       # postcode input (= |Enc(z)|)
miprime  = ⌈R·ni⌉ - ni - niprime        # postcode output
```

### Row densities (lcpc `matgen.rs`)

```
cn = min( max( ⌈ni·32β/25⌉, 4 + ⌈ni·β⌉ ),
          ⌈(110/ni + cnst_cn_1) / cnst_cn_2⌉ )          capped at mi
dn = min( ⌈ni·2β⌉ + ⌈(⌈R·ni⌉ - ni + 110) / log2(|F|)⌉,
          ⌈(110/ni + cnst_dn_1) / cnst_dn_2⌉ )          capped at miprime

cnst_cn_1 = ent(β) + α·ent(1.28β/α)
cnst_cn_2 = β·log2(α/(1.28β))
cnst_dn_1 = R·α·ent(β/R) + μ·ent(ν/μ)
cnst_dn_2 = α·β·log2(μ/ν)
μ = R - 1 - R·α      ν = β + α·β + 0.03
ent(x) = -x·log2(x) - (1-x)·log2(1-x)
```

### Code specs (α, β, R) → δ = β/R

| spec | α | β | R | δ=β/R |
|---|---|---|---|---|
| 1 | 0.1195 | 0.0284 | 1.42 | 0.0200 |
| 2 | 0.138 | 0.0444 | 1.47 | 0.0302 |
| 3 | 0.178 | 0.061 | 1.521 | 0.0401 |
| 4 | 0.2 | 0.082 | 1.64 | 0.0500 |
| 5 | 0.211 | 0.097 | 1.616 | 0.0600 |
| 6 | 0.238 | 0.1205 | 1.72 | 0.0701 |

Trade-off: higher spec → larger δ → fewer column opens, but larger rate R
(bigger codeword, more encode work). Default to spec 6 (fewest queries).

## Commit / open / verify (tensor IOPP)

- **Commit.** Reshape the `N`-coeff poly into a `rows × k` matrix; encode each
  row → `rows × ⌈R·k⌉` encoded matrix; Merkle-hash each of the `⌈R·k⌉` columns;
  root = commitment. (Keccak256 for column hashes + Fiat-Shamir, matching the
  repo's `Keccak256Transcript`.)
- **Eval at point `r`.** The MLE eval factors as a tensor `r = (r_row, r_col)`;
  `eval = eq(r_row)·M·eq(r_col)`. Opening:
  - *Proximity:* verifier sends random `γ`; prover sends `γ·M` (one combined
    row); verifier checks it is a codeword and consistent with `t` opened columns.
  - *Consistency/eval:* prover sends `eq(r_row)·M`; verifier checks against the
    same `t` columns and computes `eval = ⟨that, eq(r_col)⟩`.
- **Column opens** `t = ⌈-128 / log2(1 - δ/3)⌉` random columns (Merkle paths).
  The `δ/3` is the proximity-test effective distance (lcpc `_n_col_opens`).

## Trait-fit notes

- Group-free: `impl PCSEngineTrait<E>` **without** `E::GE: DlogGroupExt`.
- Homomorphic methods (`combine_commitments`, `combine_blinds`,
  `rerandomize_commitment`) → return `SpartanError` (unsupported); Brakedown
  commitments are Merkle roots, not additively homomorphic.
- The Pedersen-style `ck_eval`/`comm_eval`/`blind_eval` params are ignored;
  Brakedown reveals + checks the eval via the tensor opening.

## Build order (each tested before the next)

1. encoder + code-distance sanity test — **done**
2. Merkle column commit + determinism test — **done**
3. tensor eval/open + verify + roundtrip test — **done** (+ batched Merkle multiproof)
4. soundness/tamper tests — **done**; wire under plain Spartan — **blocked, see below**

## Measured results (standalone, T256 scalar field)

- **Commit** vs Hyrax MSM (`is_small`): ~3× faster single-thread vs the SNARK's
  real commit path (~10× vs a from-scratch `Hyrax::commit`); ~4–7× faster
  multi-thread at 2¹⁴–2¹⁸ after parallelizing the Merkle tree build (loses only
  at tiny 2¹²). The remaining multi-thread limiter is the `n_rows`-way (4–8) row
  encode; tree build + leaf + column hashing are fully parallel.
- **Open / verify**: open ~5–126 ms (incl. re-commit), verify ~8–37 ms across
  2¹²–2¹⁸.
- **Proof size**: ~0.43–3.3 MB (after batched Merkle paths; was 1.9–4.9 MB). At
  2¹⁶: 1.72 MB = combined rows ~1 MB + column entries ~0.47 MB + batched auth
  ~0.23 MB. Still ~1000× Hyrax's ~KB IPA proof — the inherent code-PCS tradeoff
  (fast prover, large proof).

> **UNVERIFIED CONJECTURE (do not use without analysis or a citation):**
> dropping the proximity row `w_prox` and reusing the eval combiner `eq(r)` for
> both proximity and evaluation would cut the proof to ~1.2 MB at 2¹⁶. This was
> an inference of mine, *not* a sourced technique — the `conroi/lcpc` reference
> keeps a separate uniform-random proximity row, and the standard Brakedown/
> Ligero proximity lemma is stated for a uniform combiner, not the tensor-
> structured `eq(r)` (only `log(n_rows)` degrees of freedom). It is not
> known-unsound, just not justified. Before adopting, either find/write a
> proximity bound that holds for the tensor combiner, or discard the idea.

## Integration findings (Milestone 1 conclusion)

- **MLE convention MATCHES Spartan.** `EqPolynomial::evals_from_points`
  (`src/polys/eq.rs`) iterates `r.iter().rev()` so `r[0]` is the most-significant
  index bit; our `eq_evals` agrees. A Brakedown opening would line up with
  Spartan's sumcheck point with no reversal. ✓
- **Blocker for plain-Spartan integration: the base `PCSEngineTrait` is
  Pedersen-coupled via `comm_eval`.** `spartan.rs:424–426` commits the
  evaluation value as `comm_eval_W = G^{eval_W}` and `PCS::prove`/`verify` work
  against that group commitment. Brakedown reveals the eval directly (checked by
  the tensor opening) and has no homomorphic `comm_eval`, so it does not fit the
  trait's `prove(.., comm_eval, blind_eval)` / `verify(.., comm_eval, ..)`
  surface as written.
- **Path forward:** make `PCSEngineTrait` eval-agnostic — pass the eval *value*
  instead of `comm_eval`/`blind_eval`/`ck_eval` — exactly the refactor already
  applied to `ModPCSEngineTrait` (commit `22a9be8`). After that, Brakedown's
  `open`/`verify_open` plug in directly. This is a base-trait refactor, separate
  from (and smaller than) the Mod-PCS homomorphism rewrite.

Status: standalone Brakedown PCS is **complete and benchmarked**. Trait/Spartan
integration is deferred pending the `PCSEngineTrait` eval-agnostic refactor.
