// Copyright (c) Microsoft Corporation.
// SPDX-License-Identifier: MIT
// This file is part of the Spartan2 project.
// See the LICENSE file in the project root for full license information.
// Source repository: https://github.com/Microsoft/Spartan2

//! This module provides the trait definitions for transcript functionality in the Spartan2 library.
//! Transcripts are used for Fiat-Shamir transformations to make interactive proof systems non-interactive.
use crate::{
  errors::SpartanError,
  traits::mod_engine::{SumcheckEngine, SumcheckField},
};

/// This trait allows types to implement how they want to be added to `TranscriptEngine`.
///
/// (Step 2 of Phase 2 dropped the previous `<G: Group>` marker — the parameter
/// was only used as a phantom for type-level domain separation between
/// protocols, which the `dom_sep` label already handles. Removing it lets
/// non-group-based PCSes implement the trait without inventing a placeholder
/// curve group.)
pub trait TranscriptReprTrait: Send + Sync {
  /// returns a byte representation of self to be added to the transcript
  fn to_transcript_bytes(&self) -> Vec<u8>;
}

/// This trait defines the behavior of a transcript engine compatible with Spartan
pub trait TranscriptEngineTrait<E: SumcheckEngine>: Send + Sync {
  /// Initialize the transcript with the field's modulus context.
  ///
  /// The `params` are needed so that `squeeze` can construct challenges in
  /// dynamic-modulus fields (where the modulus is a runtime value carried
  /// in `params`). For static-modulus fields `params` is `()`.
  fn new_with_params(label: &'static [u8], params: <E::Scalar as SumcheckField>::Params) -> Self
  where
    Self: Sized;

  /// Initialize the transcript. Convenience for fields whose `Params` is
  /// `Default` (i.e. all static-modulus fields, where `Params = ()`).
  fn new(label: &'static [u8]) -> Self
  where
    Self: Sized,
    <E::Scalar as SumcheckField>::Params: Default,
  {
    Self::new_with_params(label, Default::default())
  }

  /// returns a scalar element of the field as a challenge
  fn squeeze(&mut self, label: &'static [u8]) -> Result<E::Scalar, SpartanError>;

  /// absorbs any type that implements `TranscriptReprTrait` under a label
  fn absorb<T: TranscriptReprTrait>(&mut self, label: &'static [u8], o: &T);

  /// adds a domain separator
  fn dom_sep(&mut self, bytes: &'static [u8]);
}

impl<T: TranscriptReprTrait> TranscriptReprTrait for &[T] {
  fn to_transcript_bytes(&self) -> Vec<u8> {
    self
      .iter()
      .flat_map(|t| t.to_transcript_bytes())
      .collect::<Vec<u8>>()
  }
}
