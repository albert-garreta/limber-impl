//! Brakedown polynomial commitment scheme — group-free, based on a linear-time
//! expander code + Merkle column commitments (no curve / MSM).
//!
//! See `docs/brakedown_design.md` for the construction, parameters, and the
//! commit/open/verify (tensor IOPP) plan. Built standalone first (Milestone 1);
//! the Mod-PCS integration is deferred (it needs commitment homomorphism a
//! Merkle commitment lacks).

pub mod code;
