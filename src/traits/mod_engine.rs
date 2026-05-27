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
  big_num::DelayedReduction,
  traits::{Engine, PrimeFieldExt, transcript::TranscriptEngineTrait},
};
use core::{
  fmt::Debug,
  ops::{Add, AddAssign, Mul, MulAssign, Neg, Sub, SubAssign},
};
use ff::{Field, PrimeField};
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
  + Default
  + PartialEq
  + Eq
  + Serialize
  + for<'de> Deserialize<'de>
  + Add<Output = Self>
  + Sub<Output = Self>
  + Mul<Output = Self>
  + Neg<Output = Self>
  + AddAssign
  + SubAssign
  + MulAssign
  + 'static
{
  /// The additive identity.
  fn zero() -> Self;

  /// The multiplicative identity.
  fn one() -> Self;

  /// Cast a `u64` into the field. Used for small constants (e.g. round-poly
  /// interpolation) and length-derived indices.
  fn from_u64(v: u64) -> Self;

  /// Multiplicative inverse. Returns `None` if `*self == Self::zero()`.
  fn invert(&self) -> Option<Self>;

  /// Reduce raw bytes (typically from a transcript squeeze) modulo the
  /// field's modulus. Used for deriving challenges in the transcript.
  fn from_bytes_reduce(bytes: &[u8]) -> Self;
}

/// Blanket impl: any static-modulus prime field that already implements
/// `PrimeField + PrimeFieldExt` (i.e. every `Engine::Scalar` in this codebase)
/// automatically satisfies `SumcheckField`. This is the Phase-1 backward-compat
/// path — existing code keeps running without any per-type impls.
impl<F> SumcheckField for F
where
  F: PrimeField
    + PrimeFieldExt
    + Copy
    + Send
    + Sync
    + Default
    + Serialize
    + for<'de> Deserialize<'de>
    + 'static,
{
  fn zero() -> Self {
    <F as Field>::ZERO
  }
  fn one() -> Self {
    <F as Field>::ONE
  }
  fn from_u64(v: u64) -> Self {
    Self::from(v)
  }
  fn invert(&self) -> Option<Self> {
    <F as Field>::invert(self).into()
  }
  fn from_bytes_reduce(bytes: &[u8]) -> Self {
    <Self as PrimeFieldExt>::from_uniform(bytes)
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
  ///
  /// The `DelayedReduction<Self::Scalar>` bound is used by the BDDT inner-
  /// product accumulator in `sumcheck.rs`. For static-modulus fields it's the
  /// Montgomery delayed-reduction optimization; for Phase-2 `DynPrime` we'll
  /// provide a no-op impl (crypto-bigint's Montgomery already reduces
  /// efficiently per multiplication).
  type Scalar: SumcheckField + DelayedReduction<Self::Scalar>;

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

  // TODO step 2: add `type TE: TranscriptEngineTrait<Self>` once the
  // transcript trait bound is relaxed. See SumcheckEngine note above.
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
  /// (Step 2 will replace `TranscriptReprTrait` with a marker-free
  /// `TranscriptRepr` trait — keeping the group bound here for now would
  /// require `E` to expose a `GE`, which we deliberately don't, so the
  /// bound is added back in step 2 alongside the transcript-trait change.)
  type Commitment: Clone
    + Debug
    + Send
    + Sync
    + PartialEq
    + Eq
    + Serialize
    + for<'de> Deserialize<'de>;

  /// Blind / commitment randomizer.
  type Blind: Clone
    + Debug
    + Send
    + Sync
    + PartialEq
    + Eq
    + Serialize
    + for<'de> Deserialize<'de>;

  /// Evaluation-argument data sent in the proof.
  type EvaluationArgument: Clone + Debug + Send + Sync + Serialize + for<'de> Deserialize<'de>;

  // TODO step 2: add the method signatures (`setup`, `blind`, `commit`,
  // `prove`, `verify`, etc.) once `ModEngine::TE` is available. The shapes
  // will mirror `PCSEngineTrait` (see `traits/pcs.rs`) with `E::Scalar`
  // substituted in place of the curve scalar, and the transcript parameter
  // typed as `&mut E::TE`.
}
