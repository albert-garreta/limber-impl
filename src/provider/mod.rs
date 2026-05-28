// Copyright (c) Microsoft Corporation.
// SPDX-License-Identifier: MIT
// This file is part of the Spartan2 project.
// See the LICENSE file in the project root for full license information.
// Source repository: https://github.com/Microsoft/Spartan2

//! This module implements Spartan's traits using the following several different combinations

// public modules to be used as an commitment engine with Spartan
pub mod bn254;
pub mod keccak;
pub mod pasta;
pub mod pcs;
pub mod pt256;
pub mod traits;

mod msm;

use crate::{
  dyn_prime::DynPrime,
  provider::{
    bn254::types as bn254_types,
    keccak::Keccak256Transcript,
    pasta::{pallas, vesta},
    pcs::{hyrax_pc::HyraxPCS, kzh_pc::KZHPCS, trivial_modpcs::TrivialModPCS},
    pt256::{p256, t256},
  },
  traits::{Engine, mod_engine::ModEngine, mod_engine::SumcheckEngine},
};
use core::fmt::Debug;
use serde::{Deserialize, Serialize};

/// An implementation of the Spartan Engine trait with Pallas curve and Hyrax commitment scheme
#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct PallasHyraxEngine;

/// An implementation of the Spartan Engine trait with Vesta curve and Hyrax commitment scheme
#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct VestaHyraxEngine;

/// An implementation of the Spartan Engine trait with P256 curve and Hyrax commitment scheme
#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct P256HyraxEngine;

/// An implementation of the Spartan Engine trait with T256 curve and Hyrax commitment scheme
#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct T256HyraxEngine;

/// An implementation of the Spartan Engine trait with BN254 curve and Hyrax commitment scheme
#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct Bn254Engine;

impl Engine for PallasHyraxEngine {
  type Base = pallas::Base;
  type Scalar = pallas::Scalar;
  type GE = pallas::Point;
  type TE = Keccak256Transcript<Self>;
  type PCS = HyraxPCS<Self>;
}

impl Engine for VestaHyraxEngine {
  type Base = vesta::Base;
  type Scalar = vesta::Scalar;
  type GE = vesta::Point;
  type TE = Keccak256Transcript<Self>;
  type PCS = HyraxPCS<Self>;
}

impl Engine for P256HyraxEngine {
  type Base = p256::Base;
  type Scalar = p256::Scalar;
  type GE = p256::Point;
  type TE = Keccak256Transcript<Self>;
  type PCS = HyraxPCS<Self>;
}

impl Engine for T256HyraxEngine {
  type Base = t256::Base;
  type Scalar = t256::Scalar;
  type GE = t256::Point;
  type TE = Keccak256Transcript<Self>;
  type PCS = HyraxPCS<Self>;
}

impl Engine for Bn254Engine {
  type Base = bn254_types::Base;
  type Scalar = bn254_types::Scalar;
  type GE = bn254_types::Point;
  type TE = Keccak256Transcript<Self>;
  //type PCS = HyraxPCS<Self>;
  type PCS = KZHPCS<Self>;
}

// ---- ModEngine impls (Phase 2 step 6: trivial backward-compat) ----------
//
// Smoke-test the `ModEngine` / `ModPCSEngineTrait` machinery by wiring up
// the existing curve+Hyrax engines as ModEngines whose `Scalar` is just the
// curve scalar (no dynamic prime yet). Step 7 will add ModEngines with
// `Scalar = DynPrime<LIMBS>` and a Mod-PCS that bridges DynPrime ↔ curve
// scalar.

impl ModEngine for T256HyraxEngine {
  type ModPCS = TrivialModPCS<Self>;
}

/// A Phase-2 engine whose sumcheck arithmetic runs over the dynamic-prime
/// field `DynPrime<4>` (256-bit, runtime modulus). This is *not* an
/// `Engine` — it's a `SumcheckEngine` only, used to drive `sumcheck_modp`
/// over a runtime prime. (The full `ModEngine` impl, pairing it with a
/// Mod-PCS, lands in step 7b.)
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct T256DynPrimeEngine;

impl SumcheckEngine for T256DynPrimeEngine {
  type Scalar = DynPrime<4>;
  type TE = Keccak256Transcript<Self>;
}
