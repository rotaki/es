# es — external sort

This README documents how a sort run's configuration (thread counts,
per-thread buffer sizes, merge fan-in, sparse-index reserve) is determined
end-to-end, from CLI flag to engine, and how to reproduce the headline
benchmarks.

## How a run's configuration is determined

A sort needs five values:

| Value | Meaning |
| --- | --- |
| `T_gen` | run-generation threads |
| `T_merge` | merge threads |
| `rg_buf_mb` | per-thread run-gen memory (MB) |
| `merge_fanin` | global merge fan-in |
| `sparse_index_fraction` | fraction of `rg_buf_mb` / merge buffer reserved for sparse-index pages (default 0.05) |

There are two ways to populate the first four. `sparse_index_fraction`
flows the same way regardless of which path you pick.

### Path A — explicit (default; no `--use-planner`)

You supply each value directly on the CLI.

```text
   ┌─────────────────────────────────────────────────────────────┐
   │ CLI flags                                                   │
   │   --run-gen-threads N      (T_gen)                          │
   │   --merge-threads   M      (T_merge)                        │
   │   --rg-buf-mb       X      (rg_buf_mb, total per-thread)    │
   │   --merge-fanin     F      (merge_fanin)                    │
   │   --sparse-index-fraction  (default 0.05)                   │
   └─────────────────────────────────────────────────────────────┘
                              │
                              ▼
   ┌─────────────────────────────────────────────────────────────┐
   │ BenchmarkConfig                                             │
   │   (struct fields populated 1-to-1 from CLI args)            │
   └─────────────────────────────────────────────────────────────┘
                              │
                              ▼
   ┌─────────────────────────────────────────────────────────────┐
   │ BenchmarkRunner::run_single_sort                            │
   │   ExternalSorter[WithOVC]::new(T_gen, rg_buf_mb*1MiB,        │
   │                                T_merge, merge_fanin, dir)   │
   │   sorter.set_imbalance_factor(...)                          │
   │   sorter.set_partition_type(...)                            │
   │   sorter.set_discard_final_output(...)                      │
   │   sorter.set_sparse_index_fraction(SIF)   ← new             │
   └─────────────────────────────────────────────────────────────┘
                              │
                              ▼
   ┌─────────────────────────────────────────────────────────────┐
   │ Engine (run generation, per worker)                         │
   │   thread_index_budget = ceil(rg_buf_mb*1MiB · SIF)          │
   │   data_bytes          = rg_buf_mb*1MiB − thread_index_budget│
   │   IndexingInterval.with_budget_bytes(thread_index_budget)   │
   │   run_replacement_selection(scanner, sink, data_bytes)      │
   │                                                             │
   │ Engine (multi-merge)                                        │
   │   merge_memory_bytes  = T_merge · merge_fanin · page_size   │
   │   total_index_budget  = floor(merge_memory_bytes · SIF)     │
   │   per-thread budget   = total_index_budget / T_merge        │
   └─────────────────────────────────────────────────────────────┘
```

The engine is the single place that splits memory between data buffer and
sparse-index pages — the trait impls (`run_replacement_selection_*`) just
receive `data_bytes` and use it as the heap budget verbatim.

### Path B — planner (`--use-planner`)

You supply only the budget; the planner picks `T_gen`, `T_merge`,
`rg_buf_mb`, `merge_fanin` for you.

```text
   ┌─────────────────────────────────────────────────────────────┐
   │ CLI flags                                                   │
   │   --use-planner                                             │
   │   --memory-mb     M      (total memory budget)              │
   │   --max-threads   T_max                                     │
   │   --sparse-index-fraction (default 0.05)                    │
   └─────────────────────────────────────────────────────────────┘
                              │
                              ▼
   ┌─────────────────────────────────────────────────────────────┐
   │ input_provider.estimate_data_size_mb() → D                  │
   └─────────────────────────────────────────────────────────────┘
                              │
                              ▼
   ┌─────────────────────────────────────────────────────────────┐
   │ plan_resource_efficient(PlannerConfig{                      │
   │     dataset_mb=D, memory_mb=M, max_threads=T_max,           │
   │     page_size_kb=64, sparse_index_fraction=SIF,             │
   │     min_rg_buf_mb=40                                        │
   │ })                                                          │
   │                                                             │
   │   thread-bound  : T² ≤ M_run_gen · M_merge / (D · P)        │
   │     → T_gen = T_merge = T_max (capped by min_rg_buf_mb)     │
   │     → rg_buf_mb = max(min_rg_buf_mb, D / max_fanin)         │
   │   memory-bound : otherwise                                  │
   │     → T_gen reduced to honor min_rg_buf_mb                  │
   │     → T_merge = floor(threshold / T_gen)                    │
   │     → rg_buf_mb = M_run_gen / T_gen                         │
   │                                                             │
   │   merge_fanin  = floor(M_merge / (T_merge · P)) − 1         │
   │   where M_run_gen = ρ·M, M_merge = ρ·M·(1 − SIF), ρ = 0.8   │
   └─────────────────────────────────────────────────────────────┘
                              │
                              ▼
   ┌─────────────────────────────────────────────────────────────┐
   │ BenchmarkConfig populated from PlannerResult                │
   │   (rest of pipeline identical to Path A)                    │
   └─────────────────────────────────────────────────────────────┘
```

Note: in Path B the planner's `sparse_index_fraction` shapes `merge_fanin`
(by shrinking `M_merge` before fan-in math), and the **same** value is also
plumbed to `set_sparse_index_fraction` so the runtime split matches what
the planner assumed. CLIs pass one value to both via `--sparse-index-fraction`.

### Print-only mode

`--print-plan --memory-mb M --max-threads T` runs only step 1 of Path B
(estimate + plan) and prints the resulting `PlannerResult`, then exits.
Useful for inspecting what the planner would pick without running the sort.

## Reproducing benchmark results

The two main suites are:

- [scripts/gensort_sort_bench_new.sh](scripts/gensort_sort_bench_new.sh) — fixed-size 100-byte records (Sort Benchmark / GenSort).
- [scripts/kvbin_sort_bench_new.sh](scripts/kvbin_sort_bench_new.sh) — variable-size KV records under different skew profiles (`freq_key`, `heavy_key`).

### Generate datasets

GenSort 200 GiB:

```bash
./dataset_generator/generate_gensort_dataset.sh
# → ./datasets/gensort_200GiB.data
```

KVBin (auto-generated by the bench script if absent; or manually):

```bash
cargo run --release --example gen_freq_key_kvbin -- \
  --out ./datasets/freq_key.kvbin --idx ./datasets/freq_key.kvbin.idx --rows 406720388

cargo run --release --example gen_heavy_key_kvbin -- \
  --out ./datasets/heavy_key.kvbin --idx ./datasets/heavy_key.kvbin.idx --rows 63013489
```

Row counts come from [kvbin_sort_bench_new.sh:70-71](scripts/kvbin_sort_bench_new.sh#L70-L71).

### Run the suites

```bash
# Disable rclone if you don't have a 'gdrive' remote configured.
RCLONE_REMOTE="" ./scripts/gensort_sort_bench_new.sh \
  ./datasets/gensort_200GiB.data \
  logs/gensort_$(date +%Y-%m-%d)

RCLONE_REMOTE="" ./scripts/kvbin_sort_bench_new.sh \
  ./datasets \
  logs/kvbin_$(date +%Y-%m-%d)
```

The gensort script runs Exp0–Exp6: a memory-tracked single run, two cgroup
sweeps (manual + planner), buffered-I/O variants, a memory cliff, an OVC
on/off scaling sweep, and a 7×7 (T_gen, T_merge) grid.  See
[scripts/gensort_sort_bench_new.sh:303-407](scripts/gensort_sort_bench_new.sh#L303-L407).

The KVBin script iterates `{freq_key, heavy_key} × {key-only, count-balanced, size-balanced}` at 16 threads / 2 GiB.

### Prerequisites

- Linux x86_64 (Linux-only because of `systemd-run` user-scope cgroup enforcement; non-Linux silently falls back to no cgroup).
- Rust toolchain that builds edition 2024 (≥ 1.85).
- `bc`, `tee`, `ps`, `du`, `wget`, optionally `rclone`, optionally `/usr/local/sbin/clearcache3.sh`.
- ≥ 256 GiB free disk; ≥ 48 GiB RAM for the full cgroup sweep (drop the largest sizes for smaller machines).
