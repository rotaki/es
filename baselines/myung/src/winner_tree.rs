//! Winner tree (basic tournament tree) for k-way merge.
//!
//! Matches Myung Alg. 2's `initialize_tournament(run)` / `pq.pop()` / `pq.push()`
//! semantics. We deliberately use the *winner-tree* variant — not the
//! loser-tree variant from Knuth Vol. 3 §5.4.1, which is the comparison-
//! optimised version CrocSort builds on (`tree-of-losers`). The winner tree
//! still gives the textbook ⌈log₂ k⌉ comparisons per replay, just with the
//! straightforward "compare both children" recipe rather than the loser-tree
//! shortcut.
//!
//! Layout: complete binary tree padded to `next_power_of_two(k)` leaves.
//! Internal node `i` (1-indexed) stores the leaf index that wins its subtree,
//! or `NIL` if every leaf below is exhausted. Children are `2i` and `2i+1`.
//!
//! Keys are looked up via a closure (`key_at: Fn(u32) -> Option<&[u8]>`).
//! This avoids copying keys out of the run readers on every pop — the
//! closure captures the reader array immutably, and per-replay borrows live
//! only inside `compare_pick`.

pub const NIL_LEAF: u32 = u32::MAX;

pub struct WinnerTree {
    /// Number of leaves the caller cares about. Padding leaves carry NIL.
    k: usize,
    /// Padded leaf count (next power of two ≥ k).
    leaf_cap: usize,
    /// 1-indexed array of node values. Internal nodes [1, leaf_cap), leaves
    /// at [leaf_cap, 2*leaf_cap). Each value is the index of the leaf that
    /// "owns" the winner of that subtree, or `NIL_LEAF`.
    nodes: Vec<u32>,
}

impl WinnerTree {
    pub fn new(k: usize) -> Self {
        assert!(k > 0);
        let leaf_cap = k.next_power_of_two().max(1);
        WinnerTree {
            k,
            leaf_cap,
            nodes: vec![NIL_LEAF; 2 * leaf_cap],
        }
    }

    pub fn k(&self) -> usize {
        self.k
    }

    /// Initial tournament. `key_at(i)` returns leaf i's first key (or None
    /// if leaf i is empty).
    pub fn build<'a, F>(&mut self, key_at: F)
    where
        F: Fn(u32) -> Option<&'a [u8]>,
    {
        // Seed leaves.
        for i in 0..self.k {
            self.nodes[self.leaf_cap + i] = if key_at(i as u32).is_some() {
                i as u32
            } else {
                NIL_LEAF
            };
        }
        for i in self.k..self.leaf_cap {
            self.nodes[self.leaf_cap + i] = NIL_LEAF;
        }
        // Bottom-up: pick winner at every internal node. O(k).
        for node in (1..self.leaf_cap).rev() {
            self.nodes[node] =
                compare_pick(self.nodes[2 * node], self.nodes[2 * node + 1], &key_at);
        }
    }

    /// Current root winner (leaf index in [0, k)) or `NIL_LEAF` if drained.
    pub fn winner(&self) -> u32 {
        self.nodes[1]
    }

    /// Replay leaf `leaf_idx` after its key changed. The closure must reflect
    /// the new key state for that leaf (and unchanged keys for everyone else).
    pub fn replay<'a, F>(&mut self, leaf_idx: usize, has_key: bool, key_at: F)
    where
        F: Fn(u32) -> Option<&'a [u8]>,
    {
        debug_assert!(leaf_idx < self.k);
        self.nodes[self.leaf_cap + leaf_idx] = if has_key { leaf_idx as u32 } else { NIL_LEAF };
        let mut node = (self.leaf_cap + leaf_idx) >> 1;
        while node >= 1 {
            self.nodes[node] =
                compare_pick(self.nodes[2 * node], self.nodes[2 * node + 1], &key_at);
            if node == 1 {
                break;
            }
            node >>= 1;
        }
    }
}

/// Compare two candidates and return the winner (smaller key). NIL is +∞.
/// Tie-breaking returns the *smaller* leaf index — callers can exploit this
/// for stable merge by sorting their input streams so smaller leaf index =
/// smaller run_id.
fn compare_pick<'a, F>(a: u32, b: u32, key_at: &F) -> u32
where
    F: Fn(u32) -> Option<&'a [u8]>,
{
    match (a, b) {
        (NIL_LEAF, NIL_LEAF) => NIL_LEAF,
        (NIL_LEAF, _) => b,
        (_, NIL_LEAF) => a,
        _ => {
            let ka = key_at(a).expect("active leaf must have key");
            let kb = key_at(b).expect("active leaf must have key");
            if ka <= kb { a } else { b }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn three_way_merge_winner_sequence() {
        // s0: a, d, g | s1: b, c, h | s2: e, f
        let s0 = [&b"a"[..], &b"d"[..], &b"g"[..]];
        let s1 = [&b"b"[..], &b"c"[..], &b"h"[..]];
        let s2 = [&b"e"[..], &b"f"[..]];
        let mut idx = [0usize; 3];
        let len = [s0.len(), s1.len(), s2.len()];
        let streams: [&[&[u8]]; 3] = [&s0, &s1, &s2];

        let key_of = |leaf: u32, idx: &[usize; 3]| -> Option<&'static [u8]> {
            let l = leaf as usize;
            if idx[l] < len[l] {
                Some(streams[l][idx[l]])
            } else {
                None
            }
        };

        let mut tree = WinnerTree::new(3);
        tree.build(|leaf| key_of(leaf, &idx));

        let mut out = Vec::new();
        loop {
            let w = tree.winner();
            if w == NIL_LEAF {
                break;
            }
            let l = w as usize;
            out.push(streams[l][idx[l]]);
            idx[l] += 1;
            tree.replay(l, idx[l] < len[l], |leaf| key_of(leaf, &idx));
        }
        assert_eq!(
            out,
            vec![
                &b"a"[..],
                &b"b"[..],
                &b"c"[..],
                &b"d"[..],
                &b"e"[..],
                &b"f"[..],
                &b"g"[..],
                &b"h"[..]
            ]
        );
    }

    #[test]
    fn single_leaf() {
        let s = [&b"x"[..], &b"y"[..]];
        let mut i = 0usize;
        let mut tree = WinnerTree::new(1);
        tree.build(|_| if i < s.len() { Some(s[i]) } else { None });
        let mut out = Vec::new();
        loop {
            let w = tree.winner();
            if w == NIL_LEAF {
                break;
            }
            out.push(s[i]);
            i += 1;
            tree.replay(
                0,
                i < s.len(),
                |_| if i < s.len() { Some(s[i]) } else { None },
            );
        }
        assert_eq!(out, vec![&b"x"[..], &b"y"[..]]);
    }

    #[test]
    fn padded_non_power_of_two_leaves() {
        let inputs: [&[u8]; 5] = [b"5", b"3", b"1", b"4", b"2"];
        let mut consumed = [false; 5];
        let mut tree = WinnerTree::new(5);
        let key_of = |leaf: u32, c: &[bool; 5]| -> Option<&'static [u8]> {
            let l = leaf as usize;
            if !c[l] { Some(inputs[l]) } else { None }
        };
        tree.build(|leaf| key_of(leaf, &consumed));
        let mut out = Vec::new();
        for _ in 0..5 {
            let w = tree.winner() as usize;
            out.push(inputs[w]);
            consumed[w] = true;
            tree.replay(w, !consumed[w], |leaf| key_of(leaf, &consumed));
        }
        assert_eq!(
            out,
            vec![&b"1"[..], &b"2"[..], &b"3"[..], &b"4"[..], &b"5"[..]]
        );
        assert_eq!(tree.winner(), NIL_LEAF);
    }

    #[test]
    fn tie_breaks_to_smaller_leaf_index() {
        // Both leaves currently hold the same key — winner should be leaf 0
        // (paper's stability tie-break: smaller leaf index = smaller run_id).
        let mut tree = WinnerTree::new(2);
        let same: &[u8] = b"key";
        tree.build(|_| Some(same));
        assert_eq!(tree.winner(), 0);
    }
}
