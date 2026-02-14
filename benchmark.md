# Benchmarking + cross-branch comparisons

This repo contains **Rust Criterion benchmarks** (in `benches/`) and (in `biosphere-py/`) an **ASV-style Python benchmark harness**. This document focuses on **comparing performance between two git refs/branches** and on capturing:

- **Raw speed / wall-time** (Criterion, `cargo bench`)
- **Throughput** (Criterion throughput where enabled)
- **CPU time** (via OS tools like `perf stat` or `/usr/bin/time`, optionally paired with Criterion’s `--profile-time`)
- **Allocations** (via a small allocation-counting helper binary; see below)
- **Fit vs predict** operations (separate benchmarks for each operation)

## What exists today

Rust Criterion benches (all `harness = false`):

- `benches/bench_utils.rs` (argsort, sampling helpers)
- `benches/bench_tree.rs` (tree split / fit-with-sorted-samples hot path)
- `benches/bench_forest.rs` (forest `fit_predict_oob`, `n_jobs=1` vs `n_jobs=4`)

Python benchmarks (optional):

- `biosphere-py/benchmarks/` (NYC Taxi dataset based harness)
- `biosphere-py/asv.conf.json` (ASV configuration)

## Comparing Python benchmarks (ASV, optional)

If you want a “commit A vs commit B” comparison for the Python package:

- `cd biosphere-py`
- Run a range (creates/uses `.asv/`):
  - `asv run main..HEAD`
- Compare two commits/refs:
  - `asv compare main HEAD`
- (Optional) build HTML:
  - `asv publish`
  - `asv preview`

ASV can also track memory metrics if you add `mem_*` / `peakmem_*` benchmarks.

## Ground rules for meaningful comparisons

For any “branch A vs branch B” comparison, keep these constant:

- Same machine / CPU governor settings (avoid running on battery / thermal throttling)
- Same Rust toolchain (`rustc -V`), same `RUSTFLAGS`, same `cargo` version
- Same benchmark inputs (dataset sizes / parameters)
- Prefer `--noplot` for speed and less noise from report generation

Criterion runs are wall-time based by default. If you care about **CPU time**, see the CPU-time section below.

## Running Rust benchmarks (single ref)

- Run everything: `cargo bench`
- Run one bench binary: `cargo bench --bench bench_forest`
- Filter within a bench binary (substring match): `cargo bench --bench bench_forest -- forest_4_jobs`
- Save a named baseline (stored under `target/criterion/.../<baseline>`):
  - `cargo bench --bench bench_forest -- --save-baseline main`
- Compare against a baseline:
  - `cargo bench --bench bench_forest -- --baseline main`
  - If you expect some benchmarks to exist only on one side, use lenient mode:
    - `cargo bench --bench bench_forest -- --baseline-lenient main`

Tip: `cargo bench --bench bench_forest -- --help` shows all Criterion CLI options.

## Comparing two refs/branches (Rust / Criterion)

### Option A (simple): save baseline, switch branches, compare

This is the simplest approach if you’re OK with checking out another branch in your working directory:

1. On the **base** ref (e.g. `main`):
   - `cargo bench -- --noplot --save-baseline base`
2. Switch to the **candidate** ref (your feature branch).
3. Compare:
   - `cargo bench -- --noplot --baseline-lenient base`

### Option B (recommended): compare using git worktrees (no branch switching)

This avoids disturbing your current checkout by benchmarking each ref in its own worktree, then comparing.

Use `scripts/bench_compare.sh`:

- Compare `main` vs your current `HEAD`:
  - `bash scripts/bench_compare.sh main HEAD`
- Compare two named branches:
  - `bash scripts/bench_compare.sh main my-feature-branch`
- Filter (Criterion substring filter; applies inside each bench binary):
  - `bash scripts/bench_compare.sh main HEAD forest`

By default, the script compares the benches that exist on both refs:

- `bench_utils`
- `bench_tree`
- `bench_forest`

Override the list with `BENCHES` (space-separated):

- `BENCHES="bench_utils bench_tree bench_forest bench_ops" bash scripts/bench_compare.sh main HEAD`

The script stores outputs under `benchmarks/<timestamp>_<base>_vs_<new>/`.

Note: this compares **git refs/commits**. If you want to benchmark uncommitted working-tree changes, use
Option A (baseline + branch switch) or commit the changes and compare the commit hash.

## Fit vs predict operations (Rust)

Goal: have explicit, comparable benchmarks for:

- `DecisionTree::fit`
- `DecisionTree::predict`
- `RandomForest::fit`
- `RandomForest::predict`
- `RandomForest::fit_predict_oob` (already covered by `benches/bench_forest.rs`)

Bench binary: `benches/bench_ops.rs`

Example runs:

- `cargo bench --bench bench_ops -- tree_fit`
- `cargo bench --bench bench_ops -- forest_predict`

## Throughput

Criterion can display throughput (elements/sec) if benchmarks set a throughput unit.

`benches/bench_ops.rs` sets throughput (rows/sec) for its benchmarks, so Criterion’s output includes both
time/iter and throughput for the same benchmark inputs.

## Allocation comparisons (Rust)

Criterion does not natively report allocation counts. For allocations, the plan is to use a small helper
binary that installs a counting global allocator and prints allocation stats for a chosen operation.

Binary: `src/bin/alloc_report.rs`

Usage:

- Build + run:
  - `cargo run --release --bin alloc_report -- forest_fit --n 100000 --d 10 --n-estimators 100 --max-depth 4 --n-jobs 4`
- Run the same command on two refs and compare the reported counters.

Notes:

- Allocation counting adds overhead. Use it for **counts/bytes**, not for timing.
- `alloc_report` needs to exist on both refs you compare. If you’re comparing against an older branch that
  doesn’t have it yet, use an external profiler (e.g. `heaptrack` / `valgrind massif`) or temporarily
  cherry-pick the helper binary into both branches.
- For deeper allocation profiling, pair Criterion’s `--profile-time` with external tools like `heaptrack` / `valgrind massif` (Linux).

## CPU time comparisons

Because Criterion measures wall time, use OS tools when you specifically want **CPU time**:

- Linux (recommended): `perf stat`
  - Example (run one benchmark “in profiler mode” for ~5s, repeat 5 times):
    - `perf stat -r 5 cargo bench --bench bench_forest -- --profile-time 5 forest`
  - Look at `task-clock` (CPU time), plus cycles/instructions/cache-misses as needed.
- macOS: `/usr/bin/time -l` (reports user/sys time + max RSS)

Tip: `--profile-time` makes Criterion iterate without analysis/reporting, which is often better for profilers.

## Implementation checklist (this repo)

- [x] Add `scripts/bench_compare.sh` (worktree-based baseline comparison)
- [x] Add `benches/bench_ops.rs` (fit vs predict + throughput)
- [x] Add `src/bin/alloc_report.rs` (allocation counting for operations)
- [x] Update `.gitignore` for `benchmarks/` output
