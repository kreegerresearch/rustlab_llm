# Perf flywheel iteration 04: requests to rustlab

- **Measured on:** [`perf/reports/11-suite-v4.md`](../reports/11-suite-v4.md), rustlab 0.3.64, source `1c45aa6`, Apple M4 Max with 14 cores (two performance clusters of 5, one efficiency cluster of 4), 2026-11-12. Run 11 is run 10's binary and library plus suite v4's six new keys. The scaling tables and the attention breakdown come from one-off probes on the same binary. Every acceptance target uses a suite key, so `perf-check` measures it.
- **Training step now:** 36,050–36,990 tok/s (`e2e.train_step_B8`, 27.7–28.4 ms per step), 16.5× the 0.3.9 baseline. The default Sherlock run takes 2 min 3 s.
- **Status of this iteration: optional.** Iteration 03 passed the 35,000 goal (PyTorch `float64` on this CPU), and by the stopping rule the flywheel paused there. The four requests below are not needed for this repo's lessons. They are offered because each is a general `parmap` or small-array cost that any rustlab user with a parallel workload pays, and this repo can measure them precisely.
- **Goal, if taken up:** 45,000 tok/s, with success at 42,000 and the gate passing. PyTorch `float32` (55,000–62,000) stays out of reach without an `f32` mode (Deferred).
- **Previous iteration:** [`03-requests.md`](03-requests.md), outcome table filled in, plus run 03b (the batched step, not adopted).
- **Protocol:** [`docs/perf-flywheel.md`](../../docs/perf-flywheel.md). Self-check without recording:

  ```sh
  RUSTLAB=$PWD/target/release/rustlab make -C ../rustlab_llm perf-check
  ```

## What changed in the picture

After iteration 03, a single matrix product is fast: 230–350 GFLOP/s at the model's shapes. The step is now limited by what happens when **8 `parmap` workers run at once**. One window's gradient takes 12.0 ms alone and 20.0 ms inside `parmap` × 8 (`rustlab run --profile=ops perf/profile_step.rlab`: `gpt_grad_flat` 12.0 ms average when serial, `parmap:gpt_grad_flat` 20.0 ms). Measured per primitive, with the same work in every worker:

| Inside `parmap` × 8, against 1 worker | Per-worker slowdown | Probe |
|---|---:|---|
| `zeros` for one window's caches | 7.7× | `prim.alloc_window` 0.109 ms vs `par.alloc_window_B8` 0.836 ms |
| embedding scatter loop (129 row updates) | 3.8× | 0.22 → 0.84 ms |
| matrix multiply 129 × 128 × 512 | 2.7× | 74 → 203 µs per product |
| `gelu` on 129 × 512 (516 KB) | 2.2× | 2.68 → 5.89 ms per 20 calls |
| index-write loop | 1.8× | 15.4 → 28.4 ms |
| `gelu` on 129 × 8 … 129 × 128 (≤ 129 KB) | 1.0–1.3× | |
| scalar loop, no arrays | 1.15× | 10.4 → 12.0 ms |

The interpreter itself scales. Matrix multiply, large element-wise builtins and allocation do not.

**The batched step (run 03b) does not get around this.** Stacking the 8 windows into one 1032-row pass is correct, but it runs at 99 ms against 21 ms. One stream uses one of the chip's two SME units: 300–350 GFLOP/s, against 604–666 GFLOP/s summed over 8 workers. It also pays the small-array costs of R3 once per (window, head), serially. Details are in [`03-requests.md`](03-requests.md#run-03b-the-batched-step-2026-11-12-rustlab-0364-not-adopted).

## Summary

| # | Request | Attacks | Keys and targets |
|---|---|---|---|
| R1 | Matrix multiply that scales past the two SME units | `parmap:[matmul]` 56 ms of 160 ms worker CPU; 48 µs per product in workers vs 27 µs alone | `par.matmul_129x128x512_B8` 604 → ≥ 900 GFLOP/s; `parmap:[matmul]` ≤ 40 ms per 8 windows |
| R2 | Large-array builtins and allocation that do not slow down inside workers | 2.2× `gelu`, 7.7× `zeros` per worker (iteration 03's R3, not met) | `par.gelu_129x512_B8` 1,463 → ≥ 2,500 Melem/s; `par.alloc_window_B8` 0.84 → ≤ 0.3 ms |
| R3 | Lower fixed cost on small arrays | `softmax` 129 × 129 at 457 Melem/s; an attention head 92 µs, of which 22 µs is the two products | `prim.softmax_129x129` 36 → ≤ 15 µs; `prim.attn_head_129x32` 92 → ≤ 50 µs |
| R4 | A reducing `parmap` | The 8 × 811 k result matrix (50,688 KB) and its separate sum | Worker gradients summed into one accumulator; see R4 for the key |

**Estimates.** Worker CPU is 160 ms per 8 windows. R1 saves about 20 ms of it, R2 about 13 and R3 about 13, so each worker drops from 20 ms to about 14 ms. R4 removes about 2–3 ms of main-thread work. Together that is a step of about 20 ms, roughly 50,000 tok/s. These are estimates, not measurements, and the parts overlap.

## R1: matrix multiply that scales past the two SME units

**Evidence.** Summed throughput for `A * B` at 129 × 128 × 512, with the same products in every worker:

| Workers | µs per product | Summed GFLOP/s |
|---:|---:|---:|
| 1 | 74 | 229 |
| 2 | 80 | 422 |
| 4 | 130 | 521 |
| 8 | 203 | 666 |
| 14 | 316 | 749 |

Two workers nearly double the rate, and after that the gain flattens. That fits one SME unit per performance cluster, shared by every core in it.

**Problem.** With 8 workers, each product waits for a shared matrix unit, while the cores' own vector units (42–56 GFLOP/s each with the `matrixmultiply` path, measured on 0.3.16) sit idle.

**Suggested approach.** Use both kinds of unit:
- Route a product to the NEON kernel when its cluster's SME unit is busy. Try a per-cluster busy flag, or a simple cap on concurrent SME products per cluster.
- Or set a size threshold under which products run on NEON whenever more than two threads are multiplying.
- Either way, the summed rate should approach SME (≈ 2 × 350) plus NEON (≈ 6 × 50).

**Acceptance.** `par.matmul_129x128x512_B8` ≥ 900 GFLOP/s, and `prim.matmul_129x128x512` (one thread) does not regress. `parmap:[matmul]` in `rustlab run --profile perf/profile_step.rlab` ≤ 40 ms per 8 windows. Gate PASS.

## R2: large-array builtins and allocation inside workers

**Evidence.**
- `gelu` slows 2.2× per worker on a 516 KB array, but only 1.0–1.3× on arrays of 129 KB or less.
- With one worker, the 516 KB case is about 2× faster per element than the smaller ones, which suggests `gelu` threads internally above some size. Eight workers each doing that would oversubscribe the 14 cores.
- `zeros` for one window's caches (`alloc_window`, about 37 MB as 20 arrays) is 7.7× slower per worker. Iteration 03's R3 asked for ≤ 0.6 ms; it is 0.84 ms.

**Problem.** There are two candidate causes, and this repo cannot tell them apart from outside:
1. Nested threading: builtins that spawn their own threads inside `parmap` workers.
2. Fresh pages: each large buffer comes from the OS, so 8 threads contend on page faults and zeroing.

**Suggested approach.**
- Inside a `parmap` worker, run element-wise builtins on the calling thread only.
- Give each worker thread a pool of reusable large buffers, or an allocator with per-thread caches for large blocks, so a 516 KB result does not fault in fresh pages every call.
- `rustlab run --profile` with page-fault counts per worker would show which cause dominates.

**Acceptance.** `par.gelu_129x512_B8` ≥ 2,500 Melem/s, and `prim.gelu_builtin_129x512` (one thread) does not regress. `par.alloc_window_B8` ≤ 0.3 ms. Gate PASS.

## R3: lower fixed cost on small arrays

**Evidence.** One attention head (`prim.attn_head_129x32`) takes 92 µs. Broken down on the same binary:

| Part | µs |
|---|---:|
| `softmax` 129 × 129 | 37.4 |
| `S * scale + M` 129 × 129 | 13.5 |
| `A * V` 129 × 129 × 32 | 11.5 |
| `Q * K'` 129 × 32 × 129 | 10.8 |
| slices, transpose, region write | about 4 |

`softmax` on 16.6 k elements runs at 457 Melem/s, and the scale-and-mask line at about 1,200. Large element-wise chains run at about 3,400 Melem/s per operation (`gelu(X) .* X + X` on 1032 × 512 in 0.47 ms). A step calls `softmax` 144 times and runs 16 heads per window forward and backward.

**Problem.** A fixed cost per operation (allocation, dispatch, or a threading threshold set for large arrays) dominates at 129 × 129.

**Suggested approach.**
- Profile `softmax` at 129 × 129. Exp and max-subtraction can stay in one pass over each row, with a vectorized exp.
- Check whether small element-wise operations allocate more than their result.
- A fused `S * a + M` costs one pass where two operations cost two. That is general expression fusion and only worth it if it falls out of existing work.

**Acceptance.** `prim.softmax_129x129` ≤ 15 µs. `prim.attn_head_129x32` ≤ 50 µs. Gate PASS.

## R4: a reducing `parmap`

**Evidence.**
- Each step returns 8 rows of 811,009 values from the workers: 50,688 KB, the `parmap` row's `Out (KB)`.
- Each worker first packs its gradient struct into a row (`gpt_flatten_tail`, 1.25 ms per worker in `parmap`).
- The main thread then sums the rows (`flat.batch_reduce_8x811k`, 1.78 ms).
- The `parmap` row's 25.3 ms self time is nearly the whole wall time of the 8 windows.

**Problem.** The batch dimension exists only to be summed away, but it is materialized at full size first.

**Suggested approach.** A reducing form, for example `parmap(f, 1:B, "sum")`, or a separate `parsum(f, 1:B)`. It adds each worker's result into one accumulator as workers finish, so the B × n matrix never exists. Rows arrive in completion order, so the sum order changes. Either keep a fixed pairwise order, so results are deterministic, or document that the last digits can vary between runs.

**Acceptance.** When the API exists, this repo adds `par.grad_flat_sum_B8` (the same 8 windows, summed by the new form). Target: at most `par.grad_flat_B8` minus `flat.batch_reduce_8x811k`, and `e2e.train_step_B8` improves by ≥ 2 ms. Gate PASS: the reference trajectory allows last-digit changes.

## This repo's items for run 04b

These are not requests to rustlab. They are done after rustlab's changes land, so the two gains stay separate.

| Item | Evidence | Change |
|---|---|---|
| Embedding gradient as one product | The 129-iteration scatter loop in `gpt_backward` is 0.22 ms alone and 0.84 ms in workers. A one-hot 129 × 77 selector, transposed, times `dX` is 0.022 / 0.30 ms and bit-identical (max difference 0). | Replace the loop. About 0.5 ms a step. Could be done now; held back only so iteration 04's gains stay separate. |
| Gradient reduce without the slice | Carried from iteration 03: `sum(Gm(:, 1:n), 1)` copies the slice first | `sum(Gm, 1)`, then drop the loss column. Superseded if R4 lands. |
| Unflatten | `flat.unflatten_811k` 1.2 ms | Revisit after R2 |

## Deferred

| Item | Why it waits |
|---|---|
| `f32` mode | The only route to PyTorch `float32` (55,000–62,000 tok/s). Needs real-valued storage end to end and its own gate tolerance. |
| Batched multiply (`pagemtimes`) | Run 03b shows the batched step loses mainly because one stream uses one SME unit. `pagemtimes` alone would not fix that. Revisit if one large product can use both SME units. |
| One large product across both SME units | Would make the batched step worth re-measuring. Large products are not the limit today. |
| Fused causal attention with backward | R3 covers most of the per-head cost. Fused attention is a bigger API for a smaller remaining share. |

## Outcome (filled in after verification)

| # | Status | Measured | Note |
|---|---|---|---|
| R1 | open | | |
| R2 | open | | |
| R3 | open | | |
| R4 | open | | |
