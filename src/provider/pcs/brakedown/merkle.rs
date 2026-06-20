//! Minimal binary Merkle tree over `[u8; 32]` leaves (Keccak256), with
//! domain-separated leaf/node hashing and stateless path verification. Used to
//! commit the columns of a Brakedown encoded matrix.

use sha3::{Digest, Keccak256};

/// A 32-byte Keccak256 digest.
pub type Hash = [u8; 32];

/// Hash raw leaf data (domain tag `0`).
pub fn hash_leaf(data: &[u8]) -> Hash {
  let mut h = Keccak256::new();
  h.update([0u8]);
  h.update(data);
  h.finalize().into()
}

/// Hash two child digests (domain tag `1`).
fn hash_node(l: &Hash, r: &Hash) -> Hash {
  let mut h = Keccak256::new();
  h.update([1u8]);
  h.update(l);
  h.update(r);
  h.finalize().into()
}

/// A binary Merkle tree; leaves are padded to a power of two with the zero hash.
#[derive(Clone, Debug)]
pub struct MerkleTree {
  /// `layers[0]` are the (padded) leaf hashes; the last layer is the root.
  layers: Vec<Vec<Hash>>,
}

impl MerkleTree {
  /// Build a tree from already-hashed leaves (padded to a power of two).
  pub fn from_leaves(mut level: Vec<Hash>) -> Self {
    assert!(!level.is_empty(), "Merkle tree needs at least one leaf");
    level.resize(level.len().next_power_of_two(), [0u8; 32]);
    let mut layers = vec![level];
    while layers.last().unwrap().len() > 1 {
      let cur = layers.last().unwrap();
      let next: Vec<Hash> = cur.chunks(2).map(|p| hash_node(&p[0], &p[1])).collect();
      layers.push(next);
    }
    Self { layers }
  }

  /// The Merkle root.
  pub fn root(&self) -> Hash {
    *self.layers.last().unwrap().first().unwrap()
  }

  /// Authentication path (sibling hashes, bottom-up) for leaf `idx`.
  pub fn path(&self, mut idx: usize) -> Vec<Hash> {
    let mut path = Vec::with_capacity(self.layers.len() - 1);
    for layer in &self.layers[..self.layers.len() - 1] {
      path.push(layer[idx ^ 1]);
      idx >>= 1;
    }
    path
  }
}

/// Recompute the root from a leaf and its path; compare to `root`.
pub fn verify_path(root: &Hash, leaf_data: &[u8], mut idx: usize, path: &[Hash]) -> bool {
  let mut cur = hash_leaf(leaf_data);
  for sib in path {
    cur = if idx & 1 == 0 {
      hash_node(&cur, sib)
    } else {
      hash_node(sib, &cur)
    };
    idx >>= 1;
  }
  &cur == root
}

#[cfg(test)]
mod tests {
  use super::*;

  #[test]
  fn paths_verify_and_reject_tampering() {
    let n = 100usize;
    let data: Vec<Vec<u8>> = (0..n).map(|i| vec![i as u8; 7]).collect();
    let leaves: Vec<Hash> = data.iter().map(|d| hash_leaf(d)).collect();
    let tree = MerkleTree::from_leaves(leaves);
    let root = tree.root();
    for i in 0..n {
      let path = tree.path(i);
      assert!(
        verify_path(&root, &data[i], i, &path),
        "path {i} must verify"
      );
      // wrong data rejects
      assert!(!verify_path(&root, b"bogus", i, &path));
      // wrong index rejects
      assert!(!verify_path(&root, &data[i], (i + 1) % n, &path));
    }
  }
}
