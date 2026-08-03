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
    pcs::{hyrax_pc::HyraxPCS, kzh_pc::KZHPCS},
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

// ---- ModEngine impls ------------------------------------------------------

/// A Phase-2 engine whose sumcheck arithmetic runs over the dynamic-prime
/// field `DynPrime<4>` (256-bit, runtime modulus). This is *not* an
/// `Engine` — it's a `SumcheckEngine` only, used to drive `sumcheck_modp`
/// over a runtime prime. (The full `ModEngine` impl, pairing it with a
/// Mod-PCS, lands in step 7b.)
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct T256DynPrimeEngine;

/// The T256 scalar field's prime as a `FixedMontyParams<4>`. Useful when
/// you want a `DynPrime<4>` whose modulus matches the static `t256::Scalar`
/// (e.g. for `p = q` mode in tests). Computed as `(q-1) + 1` from
/// `-Scalar::ONE` to avoid parsing `PrimeField::MODULUS` string formats.
pub fn t256_scalar_params() -> crypto_bigint::modular::FixedMontyParams<4> {
  use crypto_bigint::{Odd, U256};
  use ff::{Field, PrimeField};
  let q_minus_1 = (-<pt256::t256::Scalar as Field>::ONE).to_repr();
  let mut bytes = q_minus_1.as_ref().to_vec();
  let mut carry = 1u8;
  for b in bytes.iter_mut() {
    let (v, c) = b.overflowing_add(carry);
    *b = v;
    carry = u8::from(c);
  }
  debug_assert_eq!(carry, 0, "addition carried out of the modulus width");
  let modulus = U256::from_le_slice(&bytes);
  crypto_bigint::modular::FixedMontyParams::new(Odd::new(modulus).unwrap())
}

impl SumcheckEngine for T256DynPrimeEngine {
  type Scalar = DynPrime<4>;
  type TE = Keccak256Transcript<Self>;
}

/// A `ModEngine` identical to [`T256DynPrimeEngine`] except that its
/// Mod-PCS commits with Brakedown (hash-based, non-hiding) instead of
/// Pedersen/Hyrax. Shares the same scalar field and prime sampling, so
/// the protocol layer is reused verbatim; only the commitment scheme
/// differs. This is the comparison instantiation
/// against code-commitment systems (fast prover, large proofs).
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct T256DynPrimeBdEngine;

impl SumcheckEngine for T256DynPrimeBdEngine {
  type Scalar = DynPrime<4>;
  type TE = Keccak256Transcript<Self>;
}

impl ModEngine for T256DynPrimeBdEngine {
  type ModPCS = crate::provider::pcs::integer_modpcs::IntegerModPCSBd;

  fn bootstrap_params() -> crypto_bigint::modular::FixedMontyParams<4> {
    <T256DynPrimeEngine as ModEngine>::bootstrap_params()
  }

  fn sample_params<T: crate::traits::transcript::ByteTranscript>(
    transcript: &mut T,
  ) -> crypto_bigint::modular::FixedMontyParams<4> {
    <T256DynPrimeEngine as ModEngine>::sample_params(transcript)
  }
}

impl ModEngine for T256DynPrimeEngine {
  // Phase-3 step B: sound IntegerModPCS (small-prime fingerprinting +
  // Hyrax-T256 underneath).
  type ModPCS = crate::provider::pcs::integer_modpcs::IntegerModPCS;

  /// Bootstrap params: smallest valid odd-modulus `FixedMontyParams<4>`
  /// (modulus = 3). Used only for transcript construction before the
  /// real `p` is sampled; never participates in arithmetic.
  fn bootstrap_params() -> crypto_bigint::modular::FixedMontyParams<4> {
    use crypto_bigint::{Odd, U256};
    crypto_bigint::modular::FixedMontyParams::new(Odd::new(U256::from(3u32)).unwrap())
  }

  /// Rejection-sample a ~128-bit prime `p` from the transcript via
  /// Miller-Rabin + Lucas BPSW (`crypto_primes::is_prime`). Each
  /// `squeeze_bytes` call advances the transcript identically on prover
  /// and verifier sides, so both arrive at the same `p`.
  ///
  /// The candidate is built by taking 16 bytes from the squeeze, forcing
  /// the top bit (MSB of bit 127, exactly 128-bit width) and the bottom
  /// bit (odd), then testing primality. The zero-padded `U256` carrier
  /// type matches `DynPrime<4>`'s backing.
  fn sample_params<T: crate::traits::transcript::ByteTranscript>(
    transcript: &mut T,
  ) -> crypto_bigint::modular::FixedMontyParams<4> {
    use crypto_bigint::{Odd, U256};
    use crypto_primes::{Flavor, is_prime};
    loop {
      let bytes = transcript
        .squeeze_bytes(b"sample_p")
        .expect("transcript squeeze failed during prime sampling");
      let mut candidate_bytes = [0u8; 32];
      candidate_bytes[..16].copy_from_slice(&bytes[..16]);
      candidate_bytes[15] |= 0x80; // exactly 128-bit
      candidate_bytes[0] |= 0x01; // odd
      let candidate = U256::from_le_slice(&candidate_bytes);
      if is_prime(Flavor::Any, &candidate) {
        return crypto_bigint::modular::FixedMontyParams::new(Odd::new(candidate).unwrap());
      }
    }
  }
}
