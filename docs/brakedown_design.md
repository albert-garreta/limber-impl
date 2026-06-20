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

1. encoder + code-distance sanity test ← **current**
2. Merkle column commit + determinism test
3. tensor eval/open + verify + roundtrip test
4. soundness/tamper tests + wire under plain Spartan
