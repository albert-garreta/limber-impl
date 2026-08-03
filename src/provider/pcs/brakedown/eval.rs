//! Brakedown evaluation opening — the tensor IOPP.
//!
//! The committed poly is the `n_rows × row_len` message matrix `M`. An MLE eval
//! at `r = (r_hi, r_lo)` factors as `eval = ⟨ e_row · M , e_col ⟩` with
//! `e_row = eq(r_hi)` (length `n_rows`), `e_col = eq(r_lo)` (length `row_len`).
//!
//! The prover sends two combined message rows — `w_prox = r_comb·M` for a random
//! `r_comb` (proximity / well-formedness) and `w_eval = e_row·M` (the eval) —
//! plus `t` opened columns of the encoded matrix (with Merkle paths). For each
//! opened column `c` the verifier checks, using the code's linearity
//! (`combo·EncodedMatrix = Enc(combo·M)`):
//! `Enc(w_prox)[c] == ⟨r_comb, col_c⟩` and `Enc(w_eval)[c] == ⟨e_row, col_c⟩`,
//! and finally `eval == ⟨w_eval, e_col⟩`.

use super::{
  code::{next_index, next_scalar, xof},
  commit::{BrakedownCommitData, BrakedownParams, column_to_bytes, commit},
  merkle::{Hash, hash_leaf, verify_batch_path},
};
use crate::{
  errors::SpartanError,
  traits::{PrimeFieldExt, transcript::ByteTranscript},
};
use ff::Field;
use std::collections::HashMap;

/// Evaluation argument: the proximity row, the eval row, the opened columns
/// (sorted-unique `(index, entries)`), and one batched Merkle multiproof
/// covering all of them.
#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
#[serde(bound(
  serialize = "F: serde::Serialize",
  deserialize = "F: serde::de::DeserializeOwned"
))]
pub struct BrakedownEvalArg<F> {
  w_prox: Vec<F>,
  w_eval: Vec<F>,
  columns: Vec<(usize, Vec<F>)>,
  auth: Vec<Hash>,
}

/// eq-polynomial evaluations over `r`, high-bit-first: `out[k] = ∏_b (k_b?
/// r_b : 1-r_b)` with `r[0]` the most-significant index bit. Factorizes as
/// `eq(r_hi||r_lo)[i·2^|lo| + j] = eq(r_hi)[i]·eq(r_lo)[j]`.
fn eq_evals<F: Field>(r: &[F]) -> Vec<F> {
  let mut e = vec![F::ONE];
  for &ri in r {
    let mut next = Vec::with_capacity(e.len() * 2);
    for &ev in &e {
      next.push(ev * (F::ONE - ri));
      next.push(ev * ri);
    }
    e = next;
  }
  e
}

fn inner<F: Field>(a: &[F], b: &[F]) -> F {
  a.iter().zip(b).fold(F::ZERO, |acc, (x, y)| acc + *x * *y)
}

/// `combo · M` over the systematic (message) columns: `out[j] = Σ_i combo[i] ·
/// encoded[i][j]` for `j < row_len`.
fn combine_message_rows<F: Field>(encoded: &[Vec<F>], combo: &[F], row_len: usize) -> Vec<F> {
  let mut out = vec![F::ZERO; row_len];
  for (i, row) in encoded.iter().enumerate() {
    let c = combo[i];
    for (o, &m) in out.iter_mut().zip(&row[..row_len]) {
      *o += c * m;
    }
  }
  out
}

fn expand_scalars<F: PrimeFieldExt>(seed: &[u8; 64], n: usize) -> Vec<F> {
  let mut r = xof(seed, b"sc");
  (0..n).map(|_| next_scalar::<F>(&mut r)).collect()
}

fn expand_indices(seed: &[u8; 64], count: usize, bound: usize) -> Vec<usize> {
  let mut r = xof(seed, b"idx");
  (0..count).map(|_| next_index(&mut r, bound)).collect()
}

fn verr(reason: &str) -> SpartanError {
  SpartanError::ProofVerifyError {
    reason: format!("brakedown: {reason}"),
  }
}

/// Prove `poly(point) = eval`. Re-derives the commitment internally (so the
/// caller need only hold the root); returns the evaluation and its argument.
pub fn open<F: PrimeFieldExt>(
  params: &BrakedownParams<F>,
  poly: &[F],
  point: &[F],
  transcript: &mut impl ByteTranscript,
) -> Result<(F, BrakedownEvalArg<F>), SpartanError> {
  let (root, data) = commit(params, poly);
  open_with_data(params, &root, &data, point, transcript)
}

/// Like [`open`], but against an already-committed polynomial's retained
/// [`BrakedownCommitData`] — skips the re-encode + re-hash `open` pays.
pub fn open_with_data<F: PrimeFieldExt>(
  params: &BrakedownParams<F>,
  root: &Hash,
  data: &BrakedownCommitData<F>,
  point: &[F],
  transcript: &mut impl ByteTranscript,
) -> Result<(F, BrakedownEvalArg<F>), SpartanError> {
  transcript.absorb_bytes(b"bd_root", root);
  let log_rows = params.n_rows.trailing_zeros() as usize;
  let e_row = eq_evals(&point[..log_rows]);
  let e_col = eq_evals(&point[log_rows..]);

  let seed_rc = transcript.squeeze_bytes(b"bd_rcomb")?;
  let r_comb = expand_scalars::<F>(&seed_rc, params.n_rows);
  let w_prox = combine_message_rows(&data.encoded, &r_comb, params.row_len);
  let w_eval = combine_message_rows(&data.encoded, &e_row, params.row_len);
  transcript.absorb_bytes(b"bd_wprox", &column_to_bytes(&w_prox));
  transcript.absorb_bytes(b"bd_weval", &column_to_bytes(&w_eval));
  let eval = inner(&w_eval, &e_col);

  let seed_cols = transcript.squeeze_bytes(b"bd_cols")?;
  let idxs = expand_indices(&seed_cols, params.n_col_opens, params.n_cols);
  let mut unique = idxs;
  unique.sort_unstable();
  unique.dedup();
  let columns: Vec<(usize, Vec<F>)> = unique
    .iter()
    .map(|&c| (c, data.encoded.iter().map(|row| row[c]).collect()))
    .collect();
  let auth = data.tree.batch_path(&unique);
  Ok((
    eval,
    BrakedownEvalArg {
      w_prox,
      w_eval,
      columns,
      auth,
    },
  ))
}

/// Verify that `point` opens `comm` (the Merkle root) to `eval`.
pub fn verify_open<F: PrimeFieldExt>(
  params: &BrakedownParams<F>,
  root: &Hash,
  point: &[F],
  eval: F,
  arg: &BrakedownEvalArg<F>,
  transcript: &mut impl ByteTranscript,
) -> Result<(), SpartanError> {
  transcript.absorb_bytes(b"bd_root", root);
  let log_rows = params.n_rows.trailing_zeros() as usize;
  let e_row = eq_evals(&point[..log_rows]);
  let e_col = eq_evals(&point[log_rows..]);

  let seed_rc = transcript.squeeze_bytes(b"bd_rcomb")?;
  let r_comb = expand_scalars::<F>(&seed_rc, params.n_rows);
  if arg.w_prox.len() != params.row_len || arg.w_eval.len() != params.row_len {
    return Err(verr("bad combined-row length"));
  }
  transcript.absorb_bytes(b"bd_wprox", &column_to_bytes(&arg.w_prox));
  transcript.absorb_bytes(b"bd_weval", &column_to_bytes(&arg.w_eval));

  let seed_cols = transcript.squeeze_bytes(b"bd_cols")?;
  let idxs = expand_indices(&seed_cols, params.n_col_opens, params.n_cols);
  let mut unique = idxs.clone();
  unique.sort_unstable();
  unique.dedup();

  // The opened columns must be exactly the unique challenged indices, sorted.
  if arg.columns.len() != unique.len() {
    return Err(verr("wrong number of opened columns"));
  }
  for (col, &u) in arg.columns.iter().zip(&unique) {
    if col.0 != u {
      return Err(verr("opened column index does not match challenge"));
    }
    if col.1.len() != params.n_rows {
      return Err(verr("opened column has wrong height"));
    }
  }

  // One batched Merkle multiproof over all opened columns.
  let height = params.n_cols.next_power_of_two().trailing_zeros() as usize;
  let leaves: Vec<(usize, Hash)> = arg
    .columns
    .iter()
    .map(|(i, entries)| (*i, hash_leaf(&column_to_bytes(entries))))
    .collect();
  if !verify_batch_path(root, height, &leaves, &arg.auth) {
    return Err(verr("Merkle multiproof check failed"));
  }

  // Encode the two combined rows once (verifier-side, O(row_len)).
  let enc_prox = params.code.encode(&arg.w_prox);
  let enc_eval = params.code.encode(&arg.w_eval);

  // Per-challenge checks (a duplicate challenge index reuses its column).
  let lookup: HashMap<usize, &[F]> = arg
    .columns
    .iter()
    .map(|(i, e)| (*i, e.as_slice()))
    .collect();
  for &c in &idxs {
    let entries = lookup
      .get(&c)
      .ok_or_else(|| verr("missing opened column"))?;
    if inner(&r_comb, entries) != enc_prox[c] {
      return Err(verr("proximity check failed"));
    }
    if inner(&e_row, entries) != enc_eval[c] {
      return Err(verr("eval-consistency check failed"));
    }
  }

  if inner(&arg.w_eval, &e_col) != eval {
    return Err(verr("claimed evaluation does not match"));
  }
  Ok(())
}

#[cfg(test)]
mod tests {
  use super::*;
  use crate::{
    provider::{T256HyraxEngine, keccak::Keccak256Transcript, pcs::brakedown::code::DEFAULT_SPEC},
    traits::{Engine, transcript::TranscriptEngineTrait},
  };

  type E = T256HyraxEngine;
  type F = <E as Engine>::Scalar;

  fn rand_vec(n: usize, tag: u64) -> Vec<F> {
    let mut r = xof(b"bd-eval-test", &tag.to_le_bytes());
    (0..n).map(|_| next_scalar::<F>(&mut r)).collect()
  }

  fn run(log_n: usize) {
    let n = 1usize << log_n;
    let params = BrakedownParams::<F>::new(n, DEFAULT_SPEC, 128, b"seed");
    let poly = rand_vec(n, 1);
    let point = rand_vec(log_n, 2);

    // brute-force MLE eval (same eq convention as the reshape)
    let expected = inner(&poly, &eq_evals(&point));

    let (root, _) = commit(&params, &poly);
    let mut pt = Keccak256Transcript::<E>::new(b"bd");
    let (eval, arg) = open(&params, &poly, &point, &mut pt).unwrap();
    assert_eq!(
      eval, expected,
      "protocol eval must equal MLE eval (log_n={log_n})"
    );

    let mut vt = Keccak256Transcript::<E>::new(b"bd");
    verify_open(&params, &root, &point, eval, &arg, &mut vt).unwrap();
  }

  #[test]
  fn roundtrip_eval_matches_mle() {
    for log_n in [4usize, 8, 12, 14] {
      run(log_n);
    }
  }

  #[test]
  fn verify_rejects_tampering() {
    let log_n = 12;
    let n = 1usize << log_n;
    let params = BrakedownParams::<F>::new(n, DEFAULT_SPEC, 128, b"seed");
    let poly = rand_vec(n, 1);
    let point = rand_vec(log_n, 2);
    let (root, _) = commit(&params, &poly);
    let mut pt = Keccak256Transcript::<E>::new(b"bd");
    let (eval, arg) = open(&params, &poly, &point, &mut pt).unwrap();

    // wrong claimed eval
    let mut vt = Keccak256Transcript::<E>::new(b"bd");
    assert!(verify_open(&params, &root, &point, eval + F::ONE, &arg, &mut vt).is_err());

    // tampered eval row
    let mut bad = arg.clone();
    bad.w_eval[0] += F::ONE;
    let mut vt = Keccak256Transcript::<E>::new(b"bd");
    assert!(verify_open(&params, &root, &point, eval, &bad, &mut vt).is_err());

    // tampered opened column entry
    let mut bad = arg.clone();
    bad.columns[0].1[0] += F::ONE;
    let mut vt = Keccak256Transcript::<E>::new(b"bd");
    assert!(verify_open(&params, &root, &point, eval, &bad, &mut vt).is_err());

    // wrong commitment root
    let mut wrong_root = root;
    wrong_root[0] ^= 1;
    let mut vt = Keccak256Transcript::<E>::new(b"bd");
    assert!(verify_open(&params, &wrong_root, &point, eval, &arg, &mut vt).is_err());
  }

  /// Open/verify time and proof size across sizes. Run with:
  ///   RAYON_NUM_THREADS=1 cargo test --release --lib -- --ignored --nocapture eval_bench
  /// and again without the env var.
  #[test]
  #[ignore = "benchmark; run with --release --ignored --nocapture"]
  fn eval_bench() {
    use std::time::Instant;
    println!(
      "\n== Brakedown open/verify/proof (threads={}) ==",
      rayon::current_num_threads()
    );
    for log_n in [12usize, 14, 16, 18] {
      let n = 1usize << log_n;
      let params = BrakedownParams::<F>::new(n, DEFAULT_SPEC, 128, b"seed");
      let poly = rand_vec(n, 1);
      let point = rand_vec(log_n, 2);
      let (root, _) = commit(&params, &poly);

      let mut pt = Keccak256Transcript::<E>::new(b"bd");
      let t0 = Instant::now();
      let (eval, arg) = open(&params, &poly, &point, &mut pt).unwrap();
      let open_ms = t0.elapsed().as_secs_f64() * 1e3; // incl. internal re-commit

      let mut vt = Keccak256Transcript::<E>::new(b"bd");
      let t1 = Instant::now();
      verify_open(&params, &root, &point, eval, &arg, &mut vt).unwrap();
      let ver_ms = t1.elapsed().as_secs_f64() * 1e3;

      let fb = 32usize; // bytes per field element / hash
      let rows_b = (arg.w_prox.len() + arg.w_eval.len()) * fb;
      let col_b: usize = arg.columns.iter().map(|(_, e)| e.len() * fb).sum();
      let auth_b = arg.auth.len() * 32;
      let kb = (rows_b + col_b + auth_b) as f64 / 1024.0;
      println!(
        "n=2^{log_n:<2} open(+commit) {open_ms:7.1}ms  verify {ver_ms:6.2}ms  \
         proof {kb:8.1}KB  (t={} rows={} rowlen={} cols={})",
        params.n_col_opens, params.n_rows, params.row_len, params.n_cols,
      );
    }
  }
}
