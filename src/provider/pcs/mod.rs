// Copyright (c) Microsoft Corporation.
// SPDX-License-Identifier: MIT
// This file is part of the Spartan2 project.
// See the LICENSE file in the project root for full license information.
// Source repository: https://github.com/Microsoft/Spartan2

//! This module provides implementations of polynomial commitment schemes (PCS).

// helper code for polynomial commitment schemes
pub mod ipa;

// implementations of polynomial commitment schemes
pub mod brakedown;
pub(crate) mod commit_backend;
pub mod hyrax_pc;
pub mod integer_modpcs;

/// Pre-build the deterministic Brakedown layout for a given polynomial
/// length (public code matrices; conceptually setup work). Returns the
/// column-open count for informational use.
pub fn prewarm_brakedown_params(n: usize) -> usize {
  commit_backend::bd_params::<crate::provider::pt256::t256::Scalar>(n).n_col_opens
}
