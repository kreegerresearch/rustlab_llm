# Perf flywheel iteration 02: requests to rustlab

- **Measured on:** [`perf/reports/03-suite-v2.md`](../reports/03-suite-v2.md), rustlab 0.3.10, source `1ebb762`, Apple M4 Max with 14 cores, 2026-10-18. Run 03 is the same binary as run 02 (iteration 01b), with six benchmark keys added for this iteration.
- **Training step now:** 9,542–9,967 tok/s (`e2e.train_step_B8`, 100–107 ms per step), 4.4× the 0.3.9 baseline. Goal 17,000 tok/s, a step of about 60 ms.
- **Previous iteration:** [`01-requests.md`](01-requests.md), outcome table filled in.
- **Protocol:** [`docs/perf-flywheel.md`](../../docs/perf-flywheel.md). Self-check without recording:

  ```sh
  RUSTLAB=$PWD/target/release/rustlab make -C ../rustlab_llm perf-check
  ```

## What changed in the picture

Iteration 01 removed the arithmetic bottleneck: matrix multiply is now 46–56 GFLOP/s per thread. What remains is memory traffic and interpreter work, and the two interact. One window costs 44 ms run alone (`model.grad_flat`) but about 70 ms of CPU inside a `parmap` worker. With eight workers moving memory at once, each runs about 1.5× slower. So 8 windows take 1.88× one window, where 1× would be perfect.

The nested profile of `perf/profile_step.rlab` (run 02, 8 `parmap` windows, worker CPU summed over threads):

| Share of 564 ms worker CPU | What | Profile rows |
|---:|---|---|
| ≈ 30 % | Copies and allocation | `zeros` 248 calls allocating 300 MB, 28 ms; `gpt_flatten` 54 ms producing 101 MB; `reshape` 1,008 calls moving 103 MB, 19 ms; `[index-assign]` 38 ms; `[index]` 19 ms; `[transpose]` 784 calls, 15 ms |
| ≈ 27 % | Interpreter work inside `gpt_forward` and `gpt_backward` | Self time left after the `--profile=ops` operator rows are removed |
| ≈ 27 % | Matrix multiply | `parmap:[matmul]` 1,176 calls, 150 ms |
| ≈ 16 % | Library work outside forward and backward | `gpt_grad_flat` self time and `gpt_dlogits`; this repo's item, see the last section |

Every array is stored as `Complex<f64>`, 16 bytes per number. So each of those copies moves twice the bytes a real array would, across all eight threads at once.

## Summary

| # | Request | Attacks | Keys and targets |
|---|---|---|---|
| R1 | Shape changes and concatenation without extra copies | Copy volume and the contention it causes | `flat.reshape_811k` ≤ 0.5 ms; `flat.flatten_811k` ≤ 1.5 ms; `par.grad_flat_B8` ≤ 1.3 × `model.grad_flat` |
| R2 | Less interpreter work per statement in user functions | ≈ 27 % of each window | `model.forward` ≤ 10 ms; `model.backward` ≤ 15 ms; `prim.scalar_loop_1e5` ≤ 10 ms |
| R3 | Transposed operands passed to dgemm without materializing them | 98 transposes per window | `prim.matmul_A_Bt_129x128x128` ≥ 45 GFLOP/s; no `[transpose]` rows for `X' * Y` or `X * Y'` |
| R4 | Real-valued storage: a design and a first slice | Halves the bytes of every copy | Design in rustlab's `dev/plans/`; `parmap` output of `profile_step.rlab` 101,376 → ≈ 50,688 KB |

If R1–R3 land and the library items below are done, the estimate is a step of about 55–65 ms, roughly 16,000–19,000 tok/s. The iteration is a success at 14,000 tok/s or more with the gate passing. R4 is a design-first request and does not have to land in this iteration.

## R1: shape changes and concatenation without extra copies

**Evidence.** `reshape(theta, 1584, 512)` of the 811 k-element vector takes 4.0 ms (`flat.reshape_811k`), which is a full 13 MB copy for a change of shape only. `gpt_flatten`, 14 `reshape` calls and one concatenation, takes 4.0 ms serially (`flat.flatten_811k`), and 54 ms of worker CPU over 8 windows. `gpt_unflatten` reshapes 14 slices of `theta` on every step (`flat.unflatten_811k`, 4.7 ms). Inside the workers, `reshape` is 1,008 calls moving 103 MB, and `zeros` allocates 300 MB. The serial cost of each is modest. Under 8 threads the same bytes compete for memory bandwidth and the allocator, which is the measured 1.5× inflation per window.

**Problem.** Operations that change only the shape of data copy all of it, and every copy is twice the size it needs to be.

**Suggested approach.** Make `reshape` of a vector or a contiguous slice share the existing buffer, copying only on a later write. Concatenate into one preallocated output instead of pairwise. Consider per-thread buffer reuse, or an allocator that does not serialize threads, for the large short-lived temporaries in `parmap` workers. rustlab decides what fits its rules on dependencies.

**Acceptance.** `flat.reshape_811k` ≤ 0.5 ms. `flat.flatten_811k` ≤ 1.5 ms. `flat.unflatten_811k` ≤ 2 ms. `par.grad_flat_B8` ≤ 1.3 × `model.grad_flat`, carried over from iteration 01's R3. Gate PASS.

## R2: less interpreter work per statement in user functions

**Evidence.** Serially, per window, `gpt_forward` has about 4.8 ms of self time and `gpt_backward` about 7.2 ms. That is time spent in the functions' own statements, not in builtins or matrix multiply, and it is 27 % of the 44 ms window. In the workers, the same self time is 162 ms of the 564 ms CPU. `--profile=ops` attributes about 95 ms of worker time to `[index-assign]`, `[index]`, `[transpose]` and elementwise operator rows, and the rest is statement dispatch. `prim.scalar_loop_1e5` is 13.5 ms, 7.4 M iterations/s.

**Problem.** Per-statement overhead is now a quarter of the training workload. The remaining roadmap C11 work, borrowing values instead of cloning them on reads, is the obvious candidate. rustlab's own profiler on `perf/profile_step.rlab` will show more than this file can.

**Acceptance.** `model.forward` ≤ 10 ms, from 14.8. `model.backward` ≤ 15 ms, from 22.1. `prim.scalar_loop_1e5` ≤ 10 ms. Gate PASS.

## R3: transposed operands passed to dgemm without materializing them

**Evidence.** Each window forms 98 products with a transposed operand. The backward pass has 8 per layer, for example `dW2(wf, :) = hid' * dX` and `dXn2 = dpre * W1(w, :)'`, plus 3 per head, plus 2 for the LM head: 82 for the default 4 layers and 4 heads. The forward pass adds `Q * K'` once per head and layer: 16 more. The profile's 784 `[transpose]` calls over 8 windows is exactly 98 per window, 15 ms of worker CPU, plus an allocation for each. `prim.matmul_A_Bt_129x128x128` runs at 37.6 GFLOP/s against 46.5 for the untransposed product of similar size.

**Suggested approach.** Recognize `A' * B`, `A * B'` and `A' * B'` in the evaluator and hand dgemm a transposed view. `matrixmultiply` takes arbitrary strides, so no copy is needed. Only real operands matter here; for complex ones, `'` is the conjugate transpose and must keep its meaning.

**Acceptance.** `prim.matmul_A_Bt_129x128x128` ≥ 45 GFLOP/s. `prim.matmul_At_B_512x129x512` stays ≥ 53 GFLOP/s. No `[transpose]` rows from these patterns in `rustlab run --profile=ops perf/profile_step.rlab`. Gate PASS.

## R4: real-valued storage, a design and a first slice

**Evidence.** Every remaining cost in R1 is a copy of 16-byte complex numbers holding real values. `parmap` returns 101,376 KB for 8 rows of 811,009 numbers, and `zeros` allocates 300 MB per 8 windows. Iteration 01 deferred real-valued storage as "the full fix, but a large refactor". With arithmetic now fast, it has become the largest single structural item.

**Problem.** The training workload moves twice the bytes it needs to on every allocation, copy, `parmap` transfer and checkpoint. Checkpoints are 19.5 MB for 811 k parameters with Adam state, where real storage would need 9.7 MB.

**Request.** A design in rustlab's `dev/plans/`, covering the representation, how values become complex, and the migration of builtins. Also a first slice that lands real storage for the paths this workload uses: `zeros`, `randn`, elementwise ops, matrix multiply, indexing, `reshape`, `parmap` transfer, and `save`/`load`.

**Acceptance for the first slice.** The `parmap` row of `rustlab run --profile perf/profile_step.rlab` reports about 50,688 KB out. `par.grad_flat_B8` improves by 10 % or more over this iteration's other changes. Gate PASS, with the trajectory exact or within rounding.

## This repo's items for run 02b

These are not requests to rustlab. They are this side's share of the remaining cost, done after rustlab's changes land so the two gains stay separate.

| Item | Evidence | Change in `lib/gpt.rlab` |
|---|---|---|
| Double copy of the gradient per window | `model.grad_flat` 43.9 ms against 36.9 ms for forward plus backward; the 7 ms difference is `gpt_dlogits` plus `[gpt_flatten(G), L]`, which concatenates 811 k numbers and then copies them again to append the loss | Flatten with the loss in one concatenation |
| `gpt_dlogits` scalar loop | One `row` copy and one indexed write per position, 128 per window | Subtract the one-hot targets with a single indexed write on the matrix |
| Unflatten every step | `gpt_unflatten` reshapes 14 slices per step, 4.7 ms | Revisit after R1: shared-buffer `reshape` may make it nearly free |

## Deferred

| Item | Why it waits |
|---|---|
| Batched multiply (`pagemtimes`) for the per-head loop | Matrix multiply is now efficient per thread. Re-rank after R2 lowers the loop overhead. |
| Fused causal attention with backward | Attention is about 10 % of a window. Not the largest item. |
| `f32` mode | Needs real storage (R4) first, and its own gate tolerance. |
| BLAS or Accelerate | Declined by rustlab, whose rules forbid FFI in core numerics. |

## Outcome (filled in after verification)

| # | Status | Measured | Note |
|---|---|---|---|
| R1 | open | | |
| R2 | open | | |
| R3 | open | | |
| R4 | open | | |
