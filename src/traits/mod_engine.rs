//! Phase-2 trait scaffolding for running IntMod-Spartan over a
//! runtime-determined prime field.
//!
//! The existing `Engine` trait (in `traits/mod.rs`) bakes the sumcheck field
//! into the curve via `Engine::Scalar: PrimeField + PrimeFieldBits + …`. Those
//! `ff` traits require a compile-time-known modulus (`MODULUS: &'static str`),
//! which excludes dynamic-modulus arithmetic. Phase 2 needs the SNARK
//! arithmetic to happen modulo a verifier-sampled ~128-bit prime, so this file
//! introduces a parallel trait hierarchy:
//!
//!   - [`SumcheckField`]: a field-like interface that does **not** require a
//!     static modulus. Implemented by both static-modulus curve scalars
//!     (via a blanket impl that lands in step 2) and by the runtime-modulus
//!     `DynPrime` type (Phase 2b, backed by `crypto-bigint::MontyForm`).
//!
//!   - [`SumcheckEngine`]: the minimum surface that `sumcheck.rs` needs. Both
//!     `Engine` and [`ModEngine`] implement it (via blanket impls in step 2).
//!     Loosening `impl<E: Engine> SumcheckProof<E>` to `impl<E: SumcheckEngine>`
//!     is what lets the sumcheck code run over either static or dynamic fields
//!     without algorithmic changes.
//!
//!   - [`ModEngine`]: the Phase-2 engine for `IntModSpartanSNARK`. Bundles a
//!     `SumcheckField` (the dynamic prime), a [`ModPCSEngineTrait`] (the
//!     Mod-PCS), and a transcript. **Deliberately PCS-agnostic** — does not
//!     assume the Mod-PCS sits on an elliptic-curve / Pedersen commitment;
//!     FRI-based, hash-based, or other underlying structures are also fine.
//!
//!   - [`ModPCSEngineTrait`]: PCS interface for committing polynomials over
//!     `Self::Scalar` and opening them at `Self::Scalar` points. The Phase-2+
//!     analog of `PCSEngineTrait`. Concrete impls hide their underlying
//!     machinery (curve+Pedersen, FRI, Merkle, etc.) entirely.
//!
//! This file only defines the trait shapes. Step 2 of the Phase 2 plan
//! loosens `sumcheck.rs`'s bound from `E: Engine` to `E: SumcheckEngine`,
//! drops the `Group` marker from `TranscriptReprTrait`, and adds the
//! blanket impls; step 4 adds the first `ModEngine` impl (a trivial
//! backward-compat wrapper around an existing `Engine`).

use crate::{
  errors::SpartanError,
  traits::{
    Engine, PrimeFieldExt,
    transcript::{TranscriptEngineTrait, TranscriptReprTrait},
  },
};
use core::{
  fmt::Debug,
  ops::{Add, AddAssign, Mul, MulAssign, Neg, Sub, SubAssign},
};
use ff::{Field, PrimeField};
use num_bigint::BigUint;
use serde::{Deserialize, Serialize};

/// A field-like interface suitable for sumcheck arithmetic.
///
/// Unlike `ff::PrimeField`, does **not** require a compile-time-known modulus.
/// This lets the trait be implemented by both static-modulus curve scalars
/// (Phase 1) and by `DynPrime`-style dynamic-modulus types (Phase 2+).
///
/// The trait provides only what the sumcheck and MLE code requires:
/// arithmetic, additive/multiplicative identities, `u64` casting, inversion,
/// and a way to reduce raw bytes (e.g. from a transcript squeeze) into the
/// field. It deliberately omits anything that assumes a static modulus
/// (`MODULUS`, `NUM_BITS`, `S`, etc.).
pub trait SumcheckField:
  Sized
  + Copy
  + Clone
  + Send
  + Sync
  + Debug
  + PartialEq
  + Eq
  + Add<Output = Self>
  + Sub<Output = Self>
  + Mul<Output = Self>
  + Neg<Output = Self>
  + AddAssign
  + SubAssign
  + MulAssign
  + TranscriptReprTrait
  + 'static
{
  /// Per-modulus context. For static-modulus fields this is `()` (the
  /// modulus is baked into the type at compile time). For dynamic-modulus
  /// fields like `DynPrime`, this carries the runtime modulus and the
  /// Montgomery constants derived from it.
  ///
  /// Constructors (`zero`, `one`, `from_u64`) take `&Self::Params` because
  /// for runtime moduli they need the modulus to materialize a value.
  /// Arithmetic ops between two `Self` values don't need params because
  /// dynamic-field implementors (e.g. `crypto-bigint::MontyForm`) carry
  /// the params per element internally.
  type Params: Clone + Debug + Send + Sync + 'static;

  /// The additive identity. Replaces `ff::Field::ZERO`; method form so it
  /// works for runtime-modulus fields where the const form would not be
  /// const-evaluable.
  fn zero(params: &Self::Params) -> Self;

  /// The multiplicative identity. Replaces `ff::Field::ONE`; method form
  /// for the same reason as `zero()`.
  fn one(params: &Self::Params) -> Self;

  /// Cast a `u64` into the field, reducing modulo the field's modulus.
  fn from_u64(params: &Self::Params, v: u64) -> Self;

  /// Multiplicative inverse. Returns `None` if `*self` is the additive
  /// identity (which has no inverse).
  fn invert(&self) -> Option<Self>;

  /// Serialize the field element to little-endian bytes. Used by
  /// `polys_modp` when implementing `TranscriptReprTrait` for round
  /// polynomials.
  fn to_le_bytes(&self) -> Vec<u8>;

  /// Reduce raw bytes (from a transcript squeeze) into a field element.
  /// Used to derive Fiat-Shamir challenges. For static-modulus fields this
  /// forwards to `PrimeFieldExt::from_uniform`; for dynamic-modulus fields
  /// it builds an integer from the bytes and reduces mod the modulus.
  ///
  /// NOTE (followup): the dynamic-field implementation currently consumes
  /// at most `LIMBS * 8` bytes, so for a modulus near that width the
  /// challenge is slightly biased. Fine for the Phase-2 prototype; a
  /// soundness-grade version needs wide reduction (≥ modulus_bits + 128
  /// input bits).
  fn from_bytes_reduce(params: &Self::Params, bytes: &[u8]) -> Self;
}

/// Blanket impl: any static-modulus prime field that already implements
/// `PrimeField + PrimeFieldExt` (i.e. every `Engine::Scalar` in this codebase)
/// automatically satisfies `SumcheckField`. This is the Phase-1 backward-compat
/// path — existing code keeps running without any per-type impls.
impl<F> SumcheckField for F
where
  F: PrimeField + PrimeFieldExt + TranscriptReprTrait + Copy + Send + Sync + 'static,
{
  type Params = ();

  fn zero(_: &()) -> Self {
    <F as Field>::ZERO
  }
  fn one(_: &()) -> Self {
    <F as Field>::ONE
  }
  fn from_u64(_: &(), v: u64) -> Self {
    Self::from(v)
  }
  fn invert(&self) -> Option<Self> {
    <F as Field>::invert(self).into()
  }
  fn to_le_bytes(&self) -> Vec<u8> {
    self.to_repr().as_ref().to_vec()
  }
  fn from_bytes_reduce(_: &(), bytes: &[u8]) -> Self {
    <F as PrimeFieldExt>::from_uniform(bytes)
  }
}

/// The minimum engine surface that `sumcheck.rs` and the MLE polynomial
/// operations need.
///
/// Both `Engine` and [`ModEngine`] are intended to satisfy this (via blanket
/// impls added in step 2). Loosening the bound on `SumcheckProof<E>` from
/// `E: Engine` to `E: SumcheckEngine` is what lets the same sumcheck code
/// run over the curve scalar (Phase 1) or over a dynamic prime field
/// (Phase 2).
pub trait SumcheckEngine:
  Clone + Copy + Debug + Send + Sync + Sized + Eq + PartialEq + 'static
{
  /// The field over which the sumcheck arithmetic runs. Round polynomials,
  /// challenges, sumcheck claims, and MLE evaluations all live in `Scalar`.
  type Scalar: SumcheckField;

  /// Transcript engine that supports Fiat-Shamir absorption and
  /// challenge-squeeze of `Self::Scalar` values.
  type TE: TranscriptEngineTrait<Self>;
}

/// Blanket impl: any `Engine` is a `SumcheckEngine` (its `Scalar` is the
/// curve scalar, which satisfies `SumcheckField` via the blanket impl
/// above; its `TE` is just the engine's own transcript). This is the
/// Phase-1 backward-compat path — Phase-1 SNARK code can switch its trait
/// bound from `E: Engine` to `E: SumcheckEngine` without changing any
/// concrete impl.
impl<E: Engine> SumcheckEngine for E {
  type Scalar = E::Scalar;
  type TE = E::TE;
}

/// The Phase-2 engine for `IntModSpartanSNARK`.
///
/// Pairs a `SumcheckField` (the runtime-sampled prime field) with a Mod-PCS
/// that commits integer-valued polynomials and opens them at points in
/// `Self::Scalar^n`. The underlying curve and its static-modulus PCS are
/// hidden inside [`ModEngine::Curve`] — the SNARK code never sees the curve
/// scalar directly.
///
/// For Phase 1 backward compatibility, a trivial `ModEngine` impl wraps an
/// existing `Engine` with `Scalar = Engine::Scalar` and a no-op Mod-PCS that
/// delegates to the standard PCS.
pub trait ModEngine: SumcheckEngine {
  /// The Mod-PCS. Commits polynomials whose evaluations are in
  /// `Self::Scalar` and opens them at `Self::Scalar^n` points.
  ///
  /// Deliberately PCS-agnostic: this engine does **not** expose the
  /// underlying commitment machinery (curve+Pedersen, FRI, Merkle,
  /// lattice, etc.). Each `ModPCSEngineTrait` impl knows its own
  /// internals; the SNARK code that calls it does not.
  type ModPCS: ModPCSEngineTrait<Self>;

  /// Placeholder `Params` used to construct the transcript *before* the
  /// real `p` is sampled. The transcript's typed `squeeze<F>` is never
  /// called against this placeholder — only the byte-level absorb/squeeze
  /// operations run pre-`sample_params`. After sampling, the driver calls
  /// `Keccak256Transcript::set_params` with the real `params`.
  ///
  /// For static-modulus fields, return the `Params` default (`()`). For
  /// dynamic-modulus fields, return any valid params (e.g. the smallest
  /// valid odd modulus). No default impl: requiring every `ModEngine`
  /// impl to spell this out avoids the `Params: Default` bound rippling
  /// into every driver method that touches a transcript.
  fn bootstrap_params() -> <Self::Scalar as SumcheckField>::Params;

  /// Sample the runtime modulus context for `Self::Scalar` from a
  /// `ByteTranscript`. For static-modulus fields, return the `Params`
  /// default (`()`) and absorb/squeeze nothing. For dynamic-modulus
  /// fields, run the actual rejection-sampling / primality-testing loop
  /// that drives the verifier-sampled prime `p`.
  fn sample_params<T: crate::traits::transcript::ByteTranscript>(
    transcript: &mut T,
  ) -> <Self::Scalar as SumcheckField>::Params;
}

/// PCS interface for committing polynomials whose evaluations come from a
/// dynamic-prime field, and opening them at points in that field.
///
/// Phase-2 analog of `PCSEngineTrait`. Concrete implementations handle the
/// `Self::Scalar` ↔ underlying-curve-scalar conversion internally (e.g.
/// limb-splitting in Phase 3's real Mod-PCS, or trivial casting in Phase 2's
/// placeholder).
///
/// The interface deliberately mirrors `PCSEngineTrait` so the SNARK code that
/// calls it (`IntModSpartanSNARK::prove`/`verify`) has the same shape as
/// today, just with `Self::Scalar` substituted for `E::Scalar`.
pub trait ModPCSEngineTrait<E: ModEngine>: Clone + Send + Sync {
  /// Commitment key.
  type CommitmentKey: Clone + Debug + Send + Sync + Serialize + for<'de> Deserialize<'de>;

  /// Verifier key.
  type VerifierKey: Clone + Send + Sync + Serialize + for<'de> Deserialize<'de>;

  /// Commitment value, absorbable into the transcript.
  ///
  /// `TranscriptReprTrait` is marker-free (no `<G: Group>`) since the
  /// Phase-2 transcript trait split — non-group-based PCSes can implement
  /// it without inventing a placeholder curve group.
  type Commitment: Clone
    + Debug
    + Send
    + Sync
    + PartialEq
    + Eq
    + TranscriptReprTrait
    + Serialize
    + for<'de> Deserialize<'de>;

  /// Blind / commitment randomizer.
  type Blind: Clone + Debug + Send + Sync + PartialEq + Eq + Serialize + for<'de> Deserialize<'de>;

  /// Evaluation-argument data sent in the proof.
  type EvaluationArgument: Clone + Debug + Send + Sync + Serialize + for<'de> Deserialize<'de>;

  /// Sample commitment keys for vectors of length up to `n`.
  fn setup(
    label: &'static [u8],
    n: usize,
    width: usize,
  ) -> (Self::CommitmentKey, Self::VerifierKey);

  /// Eagerly initialize any lazily-computed tables in the commitment key.
  /// Default no-op; override to match `PCSEngineTrait::precompute_ck`.
  fn precompute_ck(_ck: &Self::CommitmentKey) {}

  /// Sample a fresh blind suitable for committing a polynomial of length `n`.
  ///
  /// Mirrors `PCSEngineTrait::blind` for the Phase-2 step-6 wrapping
  /// pattern. For future hash-based / non-hiding Mod-PCS impls, this can
  /// return a unit-typed sentinel.
  fn blind(ck: &Self::CommitmentKey, n: usize) -> Self::Blind;

  /// Commit to an **integer-valued** polynomial. Each entry is a
  /// non-negative bounded integer in `BigUint` form — `p`-independent and
  /// chosen before the runtime prime `p` is sampled, so the commitment
  /// binds the integers themselves.
  fn commit(
    ck: &Self::CommitmentKey,
    v: &[BigUint],
    r: &Self::Blind,
    is_small: bool,
  ) -> Result<Self::Commitment, SpartanError>;

  /// Length / shape sanity check on a commitment.
  fn check_commitment(comm: &Self::Commitment, n: usize, width: usize) -> Result<(), SpartanError>;

  /// Prove that the integer-valued polynomial `poly` evaluates at the
  /// `Z_p` point `point` to `eval` (the canonical integer in `[0, p)`
  /// representing the `Z_p` evaluation). `comm_eval` is the commitment
  /// to `eval` from the calling SNARK driver; sound Mod-PCS impls
  /// (Phase-3 IntEval) also use `eval` directly for the integer-side
  /// consistency check.
  #[allow(clippy::too_many_arguments)]
  fn prove(
    ck: &Self::CommitmentKey,
    ck_eval: &Self::CommitmentKey,
    transcript: &mut E::TE,
    comm: &Self::Commitment,
    poly: &[BigUint],
    blind: &Self::Blind,
    point: &[E::Scalar],
    eval: &BigUint,
    comm_eval: &Self::Commitment,
    blind_eval: &Self::Blind,
  ) -> Result<Self::EvaluationArgument, SpartanError>;

  /// Verify a polynomial opening. `eval` is the canonical integer in
  /// `[0, p)` representing the claimed `Z_p` evaluation; the IntEval
  /// protocol uses it to check `int_v' ≡ eval (mod p)`.
  #[allow(clippy::too_many_arguments)]
  fn verify(
    vk: &Self::VerifierKey,
    ck_eval: &Self::CommitmentKey,
    transcript: &mut E::TE,
    comm: &Self::Commitment,
    point: &[E::Scalar],
    eval: &BigUint,
    comm_eval: &Self::Commitment,
    arg: &Self::EvaluationArgument,
  ) -> Result<(), SpartanError>;
}
