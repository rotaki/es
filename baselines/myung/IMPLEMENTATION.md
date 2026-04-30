# Myung et al. (IEEE TC 2021) Baseline — Implementation Reference

This document describes the `myung` crate at [baselines/myung/](.), a faithful
reimplementation of the NVMe external-sort framework from:

> Kihyeon Myung, Sunggon Kim, Heon Young Yeom, Jiwoong Park.
> *Efficient and Scalable External Sort Framework for NVMe SSD.*
> IEEE Transactions on Computers, Vol. 70, No. 11, November 2021.

It exists as a head-to-head baseline for the CrocSort paper (reviewer R1
required: "direct comparison against Myung et al. [61] under identical hardware
and memory budgets"). The crate lives outside `src/` so that CrocSort proper
remains untouched and Myung does not silently inherit CrocSort's algorithmic
wins (replacement selection, sparse indexes, OVC, tree-of-losers).

---

## 1. Crate layout

```
baselines/myung/
├── Cargo.toml                  workspace member; depends on `es` (path)
├── src/
│   ├── lib.rs                  module root, MyungMetadata struct
│   ├── planner.rs              Eq. 2 adaptive resource allocation
│   ├── sampler.rs              upfront 0.1% boundary sampling (GenSort + KVBin)
│   ├── arbitration.rs          rw-lock + FIFO writer semaphore
│   ├── page_buffer.rs          paged record store + sortable pointer vector
│   ├── run_formation.rs        Algorithm 1: load-sort-store, implicit shuffling
│   ├── merge.rs                Algorithm 2: range-restricted k-way merge
│   └── winner_tree.rs          basic tournament tree (NOT loser tree)
├── examples/
│   └── myung_sort_cli.rs       standalone CLI binary
└── scripts/
    ├── fio_myung_calibrate.sh  measure device interleaving floor C
    └── myung_bench.sh          memory × thread × arbitration sweep harness
```

The only change to the parent `es` crate is adding `baselines/myung` to the
root workspace `[workspace] members`. No source files in `src/sort/` or
`examples/` were touched.

---

## 2. What this baseline reuses from `es` (and what it deliberately does not)

**Reused — device-level primitives only:**

- `es::diskio::file::{SharedFd, pread_fd, pwrite_fd, file_size_fd}` — Direct
  I/O / aligned I/O wrappers
- `es::diskio::aligned_reader::AlignedReader` — block-aligned chunked reads
  (used inside `KvBinInputDirect`)
- `es::input_reader::gensort_input_direct::GenSortInputDirect` — GenSort
  format parsing (10 B key + 90 B payload, fixed)
- `es::input_reader::kvbin_input_direct::KvBinInputDirect` — KVBin format
  parsing (variable-length, sidecar `.idx` of `u64` block offsets)
- `es::SortInput` trait — pluggable record source; both inputs above
  implement it

**Deliberately NOT reused:**

| CrocSort module | Why we skip it |
|---|---|
| `es::replacement_selection::*` | Myung is load-sort-store. Replacement selection (run-expansion E≈2) would inflate run length and reduce N, indirectly satisfying Eq. 2 too easily. |
| `es::sort::core::sparse_index::*` | Myung has no sparse index — implicit shuffling at run-write time replaces the merge-time seek. |
| `es::ovc::*` | Myung compares full keys. OVC compresses runs and accelerates comparisons; a CrocSort-only contribution. |
| `es::ovc::tree_of_losers*` | Loser tree is the comparison-optimised tournament variant from Knuth Vol. 3 §5.4.1. The paper says "tournament" + `pq.pop`/`pq.push`; we use a basic winner tree to match the textbook reading without lifting CrocSort's specific optimisation. |
| `es::sort_policy_sub::*` | Myung has its own resource policy (Eq. 2). |
| Best-fit allocator from `es::replacement_selection::memory` | Designed for replacement selection's irregular slot reuse; Myung uses simple fixed-size pages. |

---

## 3. Memory management (the explicit budget question)

The user-specified memory budget `M` (bytes) is passed through the planner
unchanged and then divided equally:

```
per_thread_budget = M / effective_T
```

`effective_T` is the planner-adjusted thread count (see §6). Each worker
thread holds one independent `PageBuffer` initialised with that per-thread
budget.

### 3.1 `PageBuffer` accounting (run formation)

`PageBuffer::try_push(&key, &value)` checks **before** allocating:

```
committed_bytes = pages.len() * PAGE_SIZE          (= 64 KiB × pages)
                + refs.len()  * POINTER_SIZE       (= 8 B × pointers)

incremental = POINTER_SIZE
            + (PAGE_SIZE if a new page is required to fit the record else 0)

if committed_bytes + incremental > budget:  return false  -> caller flushes
```

So the read-step working set is hard-capped at `per_thread_budget` per thread.
Pages are counted at full `PAGE_SIZE` (not actual used bytes) — pessimistic
and faithful to "the page is committed for the life of the run."

The pointer vector cost (8 bytes per record) **is included** in the budget,
per the user's instruction to "account for that memory, too."

### 3.2 Flush-step memory (now strict)

`flush_run_to_disk` walks the sorted pointer vector once and streams each
record's raw bytes directly into a per-range `AlignedWriter` (Direct I/O,
O_DIRECT via `es::diskio::file::SharedFd`). Writers are opened lazily on
the first record that classifies into a given range — empty ranges never
create files.

```rust
let mut writers: Vec<Option<AlignedWriter>> = vec![None; num_ranges];
for r in buffer.refs() {
    let rng = classify(buffer.key_at(r));
    let w = writers[rng].get_or_insert_with(|| open_aligned_writer(&paths[rng]));
    w.write_all(buffer.raw_record_bytes(r))?;     // Direct I/O, 64 KiB buffer
}
for (rid, slot) in writers.iter_mut().enumerate() {
    if let Some(mut w) = slot.take() { w.flush()?; /* 512 B zero-padded */ }
}
```

Transient memory during flush is bounded by `R × 64 KiB` for the per-range
`AlignedBuffer`s (default internal buffer size). For `R = 16` that's 1 MiB
per thread — small and constant, does not scale with `per_thread_budget`.

The earlier version used `Vec<Vec<u8>>` staging buffers that doubled the
per-thread footprint; it has been replaced. Peak total memory during run
formation is now tightly bounded:

```
M_peak ≈ T × (per_thread_budget + R × 64 KiB)
       ≈ M + T × R × 64 KiB
```

For M = 2 GiB, T = 16, R = 16: peak overhead is 16 MiB, well under 1% of M.
No doubling any more.

### 3.3 Merge-step memory

Each merge thread holds:

- 1 **`AlignedReader`** per run file, sized per Eq. 2 — see below.
- 1 `AlignedWriter` for output, 2 MiB (Myung §3.2 MDTS).
- 1 `Vec<u8>` (`buf` inside each `RunReader`) holding the most recently
  read record header + key + value — sized exactly to the current record.
- 1 `WinnerTree` of `2k` `u32` slots = `8k` bytes for `k` input runs.

**Per-run read block sized from Eq. 2.** The CLI computes

```text
read_block_bytes = max( (M / T_merge) / R, C )   rounded up to 512 B
```

using the actual number of runs produced during run formation, and passes
the result into `MergeConfig::read_block_bytes`. Each `RunReader` opens
its per-range file via `SharedFd` (O_DIRECT) and wraps it in
`AlignedReader::from_fd_with_buffer_size(fd, read_block_bytes, 0, None)`.

So the per-thread merge footprint is:

```text
≈ k_runs_for_this_range × read_block_bytes    (AlignedReader inputs)
+ 2 MiB                                        (AlignedWriter output)
+ k_runs_for_this_range × 8 B                  (winner tree)
+ per-record record buffer (size of one record)
```

Summed across threads this approaches `M`, matching the paper's merge-memory
claim. The CLI logs the chosen value:

```text
[myung] merge read block = 174762 bytes (Eq. 2 (M/T)/R = 174762, C floor = 4096)
```

so experiments can see when the `C` floor is binding vs. when the planner
value wins.

### 3.4 Out-of-budget side allocations

**Sampling phase — the main uncharted one.** Runs single-threaded *before*
the planner-allocated `M` budget is touched, on the main thread. Peak memory
during sampling:

- **GenSort sampler** — holds `keys: Vec<Vec<u8>>` of size
  `max(sample_ratio × num_records, num_ranges × 32, 64)`. For a 200 GiB
  GenSort input at the default 0.1% ratio:

        target_keys   = 0.001 × 2.15 × 10⁹   ≈ 2.15 M entries
        per-entry     = 10 B key + 24 B Vec<u8> header (ptr/len/cap)
        peak          ≈ 73 MiB

- **KVBin sampler** — the `.idx` file is mmaped (actually `fs::read` into a
  `Vec<u8>`) entirely: for 200 GiB with 4 MiB block index that's ~400 KiB.
  Plus the same `keys: Vec<Vec<u8>>` with variable-length key bytes.

**Order-of-magnitude context.** For typical tight-`M` regimes (2–4 GiB),
sample memory is **3–5% of `M`**. For ample-`M` (16+ GiB) it's <1%. It's
released before run formation begins — `derive_range_boundaries*` returns
only `Vec<Vec<u8>>` of size `num_ranges - 1` (the final cut points).
Nothing from the internal `keys` vector survives.

**Not counted in Myung's Eq. 2 budget**, and the paper doesn't account for
it either — §3.1.1 only notes sampling is "almost negligible" in *time*
(a few seconds on a 1 TB dataset) and says nothing about memory. For the
head-to-head we can either:

1. Leave it — documented honestly, matches the paper's treatment.
2. Compact it — store sampled keys as a flat `Vec<u8>` + parallel
   `Vec<u32>` of offsets (drops the 24 B Vec-header-per-entry tax, ~85%
   reduction). Fix is local to the sampler.
3. Budget it — subtract sample memory from `M` before computing
   `per_thread_budget`. Aligns with CrocSort's treatment of sparse indexes
   (which *does* reserve 5% of `M` explicitly).

Listed as a backlog item; behaviour matches the paper today.

**Other small allocations (all negligible):**

- `RunRangeEntry` accumulator per thread — one struct per (run, range) file
  produced. Bounded by `total_runs × num_ranges`. Bytes per entry ≈ 80.
- `pending: Option<(Vec<u8>, Vec<u8>)>` — one record buffered between
  read-step iterations. Bounded by `max_record_size`.
- `boundary_keys: Vec<Vec<u8>>` handed from sampler to run formation —
  `num_ranges - 1` entries, typically <1 KB total.
- Thread stacks, `DeviceArbiter`'s mutex/condvar, Rust runtime overhead.

### 3.5 Summary

| Bound | Holds where? | Strict? | Comment |
|---|---|---|---|
| Per-thread page buffer ≤ `M/T` | run-formation read step | **Yes** | `PageBuffer::try_push` gates every insert |
| Pointer vector counted | run-formation read step | **Yes** | 8 B × refs in `committed_bytes` |
| Total ≤ `M + T·R·64 KiB` | run-formation overall | **Yes (post-refactor)** | `AlignedWriter` per-range buffers replace the old doubling staging (§3.2) |
| Per-thread merge ≤ `M/T` | merge step | **Yes (post-refactor)** | `AlignedReader` per run sized from Eq. 2 `(M/T_merge)/R`, floored at C; 2 MiB MDTS AlignedWriter output (§3.3) |
| Sampler ≤ `M` | sampling phase | **Not enforced** | single-threaded, peaks at ~3–5% of `M` on 200 GiB + 0.1% ratio (§3.4) |

The read step, flush, and merge are now all strictly bounded and faithful
to the paper. The only remaining approximation is the sampler's `Vec<Vec<u8>>`
key storage (not counted against `M`), which tracks Myung's own treatment
(the paper calls sampling overhead "almost negligible" without further
accounting).

---

## 4. Run formation pipeline (Algorithm 1)

Per-thread loop:

```
loop {
    READ STEP   (shared lock):    pull records from input scanner into PageBuffer
                                   until try_push returns false (budget hit) OR
                                   scanner EOFs.
    SORT STEP   (no lock):        pdqsort the pointer vector, comparing by
                                   key dereference into the page buffer.
    WRITE STEP  (exclusive lock): walk sorted pointers, classify each into
                                   one of R ranges, append raw record bytes
                                   to that range's staging buffer, then
                                   write each range's buffer to a separate
                                   per-(run, range) file. Increment run_id.
    if scanner exhausted and pending is empty:  break
    else:                                        clear PageBuffer and continue
}
```

A few concrete points:

- **Input partitioning** uses the `SortInput::create_parallel_scanners` API
  to produce `T` independent iterators. The scanner implementation
  (`GenSortInputDirect`, `KvBinInputDirect`) decides how to split bytes —
  GenSort splits on record-aligned boundaries; KVBin splits on sparse
  index entries.
- **Lookahead handling**: if `try_push` rejects a record (budget exhausted),
  it is stashed in `pending` and replayed at the start of the next read
  step after the buffer has been flushed and cleared.
- **In-memory sort**: `Vec<RecordRef>::sort_unstable_by` (pdqsort). The
  paper does not specify the in-memory algorithm beyond `sort_buffer()` and
  citing Knuth + AlphaSort (a key/pointer quicksort variant). pdqsort is a
  reasonable modern reading.
- **Pointer-only sort**: `RecordRef = { page_idx: u32, offset_in_page: u32 }`
  is 8 bytes, regardless of record size. Sort moves these 8-byte pointers
  only; record bytes never move. Matches AlphaSort §[8] in spirit.
- **Stability inside a run is not preserved** (pdqsort is unstable), matching
  the paper's note that "the merging phase presented in this paper produces
  an unstable sort." Cross-run stability is preserved at merge time by the
  winner tree's leaf-index tie-break (§8).
- **Run id** is assigned globally via `AtomicU32::fetch_add(1, SeqCst)`
  inside the exclusive write lock so the count is monotone and matches the
  on-disk `run_NNNNNN_range_NNN.bin` naming.

---

## 5. Page buffer (`page_buffer.rs`)

### 5.1 Page layout

64 KiB pages, in-memory record encoding:

```
[u32 key_len LE][u32 value_len LE][key bytes][value bytes]
```

Self-describing — no separate metadata required. Records up to
`PAGE_SIZE - 8 = 65 528` bytes fit. Larger records error out (the user-asked
fix would be to allow records to span pages, but the paper's 128 B fixed
records and TPC-H lineitem rows are all well under this limit).

### 5.2 Pointer

```rust
pub struct RecordRef {
    pub page_idx:        u32,   // index into Vec<Page>
    pub offset_in_page:  u32,   // byte offset of the record header
}
```

Always 8 bytes. The pointer points to the start of the 8-byte record header,
so dereferencing recovers `(klen, vlen, key, value)` without ambiguity.

### 5.3 Operations

- `try_push(key, value)` — see §3.1
- `key_at(&RecordRef) -> &[u8]` — zero-copy key view
- `kv_at(&RecordRef) -> (&[u8], &[u8])` — zero-copy key + value
- `raw_record_bytes(&RecordRef) -> &[u8]` — full on-disk-format bytes
- `sort()` — pdqsort the pointer vector by `key_at` comparison
- `clear()` — drop pages and pointers (reset for next run)

---

## 6. Planner (`planner.rs`) — Eq. 2 adaptive allocation

Equation 2 from Myung §3.2 derived precisely:

```
1. Total runs: R = T · D / M = D / (M/T)        (load-sort-store, E=1)
2. Merge read-block size: b = (M/T) / R
3. Constraint: b ≥ C  (interleaving floor measured by fio)

⇒ M/T ≥ √(C·D)   ⇔   M ≥ T · √(C·D)   ⇔   M² ≥ C·D·T²
```

`MyungPlan::new(D, M, T_user, C)` returns:

- `threads = min(T_user, ⌊ M / √(C·D) ⌋)` — reduces T uniformly when
  infeasible (single thread pool for both phases, per the paper)
- `initially_feasible: bool` — was T_user already valid? (logged for
  benchmark transparency)

Worked numerical examples and unit tests are in `planner.rs` covering
`(D, M, T, C) ∈ {(200 GiB, 8 GiB, 16, 64 KiB), (200 GiB, 2 GiB, 16, 64 KiB),
(200 GiB, 2 GiB, 32, 64 KiB), (200 GiB, 1 GiB, 16, 64 KiB)}`. The first two
are feasible at the requested T; the latter two are reduced to T=18 and T=9
respectively.

---

## 7. Sampler (`sampler.rs`)

### 7.1 GenSort variant — `derive_range_boundaries`

Used when records are fixed-width and packed without delimiters
(paper §4.1's 128 B Indy format). Reads aligned 4 KiB pages at random
offsets via `pread_fd`, slices out fixed-`record_size` records, extracts
`key_len` bytes from each, sorts the collected keys, and picks
`num_ranges - 1` quantile cuts.

Default sample ratio: 0.001 (0.1% of records, matching Myung §3.1.1).

### 7.2 KVBin variant — `derive_range_boundaries_kvbin`

For variable-length records the sampler uses the sidecar `.idx` file —
a stream of `u64` block offsets that anchor the data file every ~4 MiB.
Pipeline:

1. Read all anchor offsets from `.idx`.
2. Probe one anchor to estimate average record size.
3. Compute `target_keys = max(sample_ratio × est_records, num_ranges × 32, 64)`.
4. Compute `probes_needed = ⌈target_keys / records_per_anchor⌉`,
   capped by available anchor count.
5. Random subsample of anchors (without replacement) via
   `IndexedRandom::choose_multiple`.
6. For each chosen anchor: read a 64 KiB window with one `pread`, parse
   length-prefixed records from the start, collect up to
   `records_per_anchor` keys.
7. Sort keys, pick quantile cuts.

This honours the user's "you can further sample from them to provide ~0.1%"
guidance: the index gives us coarse anchor positions, then we sub-sample
records from those.

### 7.3 Boundary classification — `classify(key, &boundaries)`

Binary search:

- `boundaries[i]` is the **exclusive** upper bound of range `i`
- `key < boundaries[0]` → range 0
- `key == boundaries[i]` → range `i + 1` (pushes equal keys to the upper
  range — deterministic and what the paper implies)
- `key ≥ boundaries[last]` → final range

**Known weakness inherited from the paper:** if the sampler picks two
identical boundary keys (likely under heavy duplicates), the range between
them is empty by construction. CrocSort solves this via virtual total
order `(key, run_id, offset)`; we deliberately do not, because that would
remove a key Myung-vs-CrocSort comparison point.

---

## 8. Arbitration (`arbitration.rs`)

The reader-writer lock + FIFO writer semaphore from Myung §3.3.

**Semantics:**

- Many concurrent readers OR one writer (standard rw-lock).
- A writer that calls `write()` takes a FIFO ticket; it blocks until
  (a) all current readers drain, (b) no other writer is active, AND
  (c) it is the next ticket due.
- Readers and writers are mutually exclusive.

**Why this exists (paper §4.2.1):** the NVMe driver bypasses the kernel
I/O scheduler, so the SSD controller is free to reorder concurrent writes
and prioritise reads — leading to severe write-side starvation under
contention. Serialising writes collapses the variance: each writer's batch
finishes promptly within its slot, so the slowest-finisher (makespan)
improves even though aggregate write bandwidth does not.

**Toggle:** `DeviceArbiter::new(false)` produces a no-op arbiter that
returns immediately from both `read()` and `write()`. This is the on/off
A/B switch (`--no-arbitration` on the CLI) used to reproduce the paper's
34% makespan claim — or measure whatever value the testbed actually shows.

**Tested:** unit test asserts `max_concurrent_writers == 1` under stress
(8 threads alternating reads/writes for 80 iterations).

---

## 9. Merge (`merge.rs`) — Algorithm 2

### 9.1 Per-thread responsibility

Each merge thread is assigned one or more ranges (round-robin: range
`i` → thread `i mod T_merge`). The thread merges across all per-(run, range)
files for each of its ranges.

If `T_merge ≥ num_ranges`, each thread owns exactly one range and produces
exactly one output file.

### 9.2 No seeking inside run files

Implicit shuffling at run formation has already separated records by range
into different files. The merge thread opens only its range's files and
scans each linearly — no sparse-index seeks, no in-file binary search.

### 9.3 Reader

Each `RunReader` wraps a 256 KiB `BufReader`. `advance()` reads:

```
[u32 klen][u32 vlen][key bytes][value bytes]
```

into a heap `Vec<u8>` sized exactly to the record (resized on demand).
The record's bytes remain valid until the next `advance()` on that
reader — which means after popping a winner, we can emit
`current_record_bytes()` *before* advancing, and the `WinnerTree::replay`
that follows reads `current_key()` from a fresh record.

### 9.4 Winner tree (`winner_tree.rs`)

Basic tournament tree:

- `k` leaves (padded to next power of two with NIL).
- Internal node `i` (1-indexed) stores the index of the leaf that wins
  its subtree, or `NIL_LEAF`.
- `build(key_at)`: bottom-up, O(k) compares.
- `winner() -> u32`: O(1) — root.
- `replay(leaf_idx, has_key, key_at)`: ⌈log₂ k⌉ compares. Walks
  leaf-to-root, recomputing winners along the path.
- `compare_pick(a, b, key_at)`: ties break to the *smaller* leaf index.

**Stability:** because run files are opened in ascending `run_id` order,
leaf index 0 ↔ smallest run_id. Tie-break-to-smaller-leaf therefore
implements the paper's `stable_priority_queue` semantics structurally,
without packing run_id into the comparison key.

**Why winner tree, not loser tree:** the paper's pseudocode says
`pq.pop()` / `pq.push()` and the prose says "tournament sort." Loser tree
is Knuth's specific comparison-optimisation that CrocSort's tree-of-losers
implementation builds on. To avoid silently giving Myung CrocSort's
optimisation, we use the textbook winner-tree variant. Winner tree still
gives ⌈log₂ k⌉ compares per replay — just with the simple "compare both
children" recipe instead of the loser-tree shortcut.

### 9.5 Output

Per-thread Direct I/O `AlignedWriter` with a 2 MiB internal buffer,
matching NVMe MDTS (Myung §3.2: "We statically allocated 2 MB for each
merge buffer, which corresponds to max data transfer size (MDTS) of the
NVMe SSD"). The writer issues one aligned `pwrite` per full 2 MiB buffer.

Direct I/O rounds each file up to the next 512 B boundary; we call
`ftruncate` after `flush()` to trim the file back to the logical record
byte count so downstream consumers see no trailing zeros. Implementation
in `merge.rs` uses `libc::ftruncate` on `AlignedWriter::get_fd()`.

Output files are named `merged_t{TID:03}_r{RANGE:03}.bin`. Concatenating
them in `(range_id ascending, tid ascending)` order recovers the global
sorted stream.

---

## 10. On-disk formats

**Per-(run, range) intermediate files** and **merge output files** both use
the same length-prefixed record format:

```
[u32 klen LE][u32 vlen LE][key bytes][value bytes]
```

This format is self-describing for variable-length records. It differs
from the GenSort native format (raw 100-byte records, no headers) and
from the KVBin format (`[u32 klen][key][u32 vlen][value]` — interleaved,
not header-prefix). For GenSort comparison against `valsort` we strip the
8-byte headers in a small Python post-processing step (see §11).

The `MyungMetadata` struct passed from run formation to merge:

```rust
pub struct MyungMetadata {
    pub num_runs:      u32,
    pub num_ranges:    u32,
    pub record_size:   usize,         // 0 for variable-length inputs
    pub boundary_keys: Vec<Vec<u8>>,  // num_ranges - 1 cuts
    pub run_ranges:    Vec<RunRangeEntry>,  // (run_id, range_id, path, ...)
}
```

---

## 11. CLI (`examples/myung_sort_cli.rs`)

```
myung_sort_cli
  --input  <PATH>           input file
  --dir    <PATH>            scratch dir for per-(run, range) files
  --output <PATH>            output dir (gets one merged_*.bin per (thread, range))
  --threads <N>              user-requested thread count (planner may reduce)
  --memory-mb <F>            total memory budget (cgroup-aligned)
  --num-ranges <N>           merge ranges (default = effective T)
  --interleave-floor-bytes <C>   from fio calibration (default 64 KiB)
  --sample-ratio <F>         default 0.001 (0.1%)
  --format <gensort|kvbin>   record parser
  --idx <PATH>               sidecar idx (required for kvbin)
  --sample-records-per-anchor <N>   for kvbin sampler (default 8)
  --no-arbitration           disable the rw-lock for fidelity ablation
  --discard-final-output     skip materialising merged output
```

Final stdout line is parseable by `scripts/myung_bench.sh`:

```
myung_sort: D=<bytes>, M=<MiB>, T(user)=<U>, T(eff)=<E>,
            runs=<R>, ranges=<G>, total=<ms> (rf=<ms>, merge=<ms>)
```

---

## 12. Verification

### 12.1 Unit tests

15 tests pass (`cargo test -p myung`):

- `planner` × 4: configs A–D from the plan numerical examples
- `page_buffer` × 3: roundtrip (fixed records), variable-length, budget enforcement
- `sampler` × 1: classify boundary semantics
- `arbitration` × 2: disabled-is-no-op, max-concurrent-writers == 1
- `winner_tree` × 4: 3-way merge, single-leaf, non-power-of-two padding,
  smaller-leaf-wins tie break

### 12.2 End-to-end smoke tests

Both pass with checksums matching across runs:

- **GenSort 100 MB**, 1M records, multiple `(M, T, num_ranges)` configs.
  Output concatenated → 8-byte headers stripped → `valsort`:

      Records: 1000000   Checksum: 7a0e0d4868cd4
      Duplicate keys: 0
      SUCCESS - all records are in order

- **KVBin 16 MB FreqKey** (200K rows, 50% heavy-hitter at key=0).
  All 200K records present and sorted within and across files. Two ranges
  empty due to boundary collision on key=0 — *expected* Myung weakness
  on heavy-duplicate workloads (the very case CrocSort's byte-balanced
  partitioning addresses).

### 12.3 Calibration script

`baselines/myung/scripts/fio_myung_calibrate.sh <file>` reproduces Myung
§3.2's exact fio config (`ioengine=psync, direct=1, rw=read,
rand_repeat=0, norandommap`) across `bs ∈ {4k..256k}`. The operator picks
the smallest `bs` at which read bandwidth saturates 80% of peak and plugs
that value into `--interleave-floor-bytes`. **Not yet run on the Intel
760p.**

---

## 13. Deviations from the paper — explicit list

These are noted both for transparency in the head-to-head and as the
"if the comparison hinges on it, fix" backlog:

1. **Sampler memory not counted in `M`.** Peaks at ~3–5% of `M` on 200 GiB
   + 0.1% ratio due to `Vec<Vec<u8>>` per-entry overhead (§3.4). Matches
   the paper's treatment (§3.1.1 does not account for sample memory
   either). Fix: compact to flat `Vec<u8>` + offsets, or budget sample
   memory against `M`.
2. **Sampler uses synchronous `pread` per anchor.** Paper §3.1.1 cites
   `aio_read()` for async concurrent anchor reads. Affects only sampling
   wall-clock (paper already calls it "almost negligible").
3. **Records on disk are length-prefixed** (`[klen][vlen][key][value]`)
   rather than raw GenSort format (paper §4.1). Necessary for variable-
   length support; harmless for sort correctness; `valsort` uses stripped
   raw output generated by a small post-processing step.
4. **In-memory sort algorithm:** pdqsort. Paper does not specify (cites
   Knuth Vol. 3 and AlphaSort as background only).
5. **Single-NVMe assumption**, no multi-device proportional allocation
   from paper §3.2's "If one storage device has double the bandwidth..."
   The testbed has one NVMe; CrocSort makes the same assumption.
6. **No thread-to-core pinning / NVMe queue-pair affinity.** Paper relies
   on NVMe's per-core queue pairs implicitly; we rely on the OS-default
   queue assignment. Typically fine on modern Linux.
7. **`MyungMetadata.record_size = 0` for variable-length inputs** is a
   sentinel; consumers should not interpret it as "0 bytes."

---

## 14. What this baseline is for

R1 from the PVLDB review: *"Please add a direct comparison against Myung
et al. [61] under identical hardware and memory budgets."*

The baseline is **not** here to win on every workload — it is here to
substantiate CrocSort's NVMe positioning by:

1. Showing Myung's strongest claim (Eq. 2 + arbitration) holds on the
   testbed (uniform GenSort, ample memory).
2. Showing where Myung breaks (heavy-hitter duplicates, payload-size skew,
   tight-`M` regimes that force T-reduction) — CrocSort's complementary
   strengths.
3. Letting the reviewer see the comparison was real, not a paper exercise.

The plan file at `~/.claude/plans/i-have-submitted-the-floofy-newt.md`
contains the full distinction matrix, head-to-head experiment design,
and paper-revision plan that this baseline supports.
