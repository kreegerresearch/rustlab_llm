# Perf flywheel iteration 03: requests to rustlab

- **Measured on:** [`perf/reports/08-suite-v3.md`](../reports/08-suite-v3.md), rustlab 0.3.11, source `09cf101` (release `78b6c49`), Apple M4 Max with 14 cores (10 performance, 4 efficiency), 2026-10-20. Run 08 is the 0.3.11 binary with this repo's iteration 02 library items, and suite v3's five new keys.
- **Revised 2026-10-23,** before rustlab started on it, after this repo measured PyTorch on the same machine ([below](#how-far-from-pytorch)) and checked 0.3.16 ([run 09](../reports/09-v0.3.16.md)). The PyTorch numbers moved matrix multiply to the top. Implicit expansion dropped from the ranked four to an optional item.
- **Training step now:** 18,660–19,310 tok/s (`e2e.train_step_B8`, 53–55 ms per step), 8.6× the 0.3.9 baseline.
- **Goal, revised:** the first goal of 17,000 tok/s was described as "roughly PyTorch on the same CPU", an estimate made before PyTorch was measured. Measured, PyTorch does the same step in float64 at 34,300–37,200 tok/s. The goal is now **35,000 tok/s, parity with PyTorch in float64 on this CPU**. **This iteration is a success at 26,000 tok/s** with the gate passing.
- **Previous iteration:** [`02-requests.md`](02-requests.md), outcome table filled in.
- **Protocol:** [`docs/perf-flywheel.md`](../../docs/perf-flywheel.md). Self-check without recording:

  ```sh
  RUSTLAB=$PWD/target/release/rustlab make -C ../rustlab_llm perf-check
  make -C ../rustlab_llm perf-torch      # the PyTorch yardstick, same machine
  ```

## How far from PyTorch

`perf/torch_twin.py` (`make perf-torch`) is the same model and step in PyTorch 2.14.1:
- V 77, d 128, 4 layers, 4 heads, d_ff 512, 811,008 parameters.
- Pre-LN, tanh GELU, fixed sinusoidal positions, untied output head.
- B = 8 windows of 129 tokens, global-norm clip 1.0, AdamW.

It ran alternately with rustlab, under the same background load, on 2026-10-23:

| Implementation | ms per step | tok/s | rustlab is |
|---|---:|---:|---:|
| rustlab 0.3.16, `float64` as complex, 8 `parmap` workers | 53.98–54.87 | 18,660–18,970 | 1× |
| PyTorch, CPU, `float64`, 1 thread | 52.6–56.5 | 18,140–19,450 | the same |
| PyTorch, CPU, `float64`, 10 threads | 27.5–29.9 | 34,300–37,190 | **1.8–2.0× slower** |
| PyTorch, CPU, `float64`, 14 threads | 31.8–32.7 | 31,300–32,180 | (efficiency cores slow it) |
| PyTorch, CPU, `float32`, 10 threads | 18.7–19.4 | 52,700–54,830 | 2.8–2.9× slower |
| PyTorch, CPU, `float32`, fused attention (SDPA) | 16.5 | 61,950–62,110 | 3.3× slower |
| PyTorch, CPU, `float32`, SDPA, `torch.compile` | 18.1–18.3 | 56,070–56,550 | 3.0× slower |
| PyTorch, GPU (MPS), `float32`, SDPA | 5.75–5.93 | 172,700–178,000 | 9.1–9.4× slower |

Like for like (`float64`, CPU), rustlab on 14 cores matches PyTorch on one thread, and is about half the speed of PyTorch on all its threads.

**Where PyTorch's advantage comes from.** Its CPU build uses Apple Accelerate (`BLAS_INFO=accelerate`), which runs matrix products on the M4's matrix unit (SME):

| Matrix multiply, `float64` | rustlab 0.3.16 | PyTorch, 1 thread | PyTorch, 10 threads |
|---|---:|---:|---:|
| 512 × 512 × 512 | 51.8 GFLOP/s | 810 | 814 |
| 129 × 128 × 512, one window's FFN | 46.1 | 351 | 353 |
| 1032 × 128 × 512, 8 windows stacked | 50.9 | 696 | 713 |
| 129 × 128 × 128 | 42.2 | 327 | 327 |

PyTorch's matrix multiply does not scale with threads, because the matrix unit is shared. rustlab's runs on one core's vector units at 1/7 to 1/16 of that rate. This CPU reports `FEAT_SME`, `FEAT_SME2` and `FEAT_SME_F64F64` (`sysctl hw.optional.arm`), with a 512-bit streaming vector length. So `float64` outer products run in hardware.

Per step, the CPU time splits like this. rustlab's figures are from run 08's profile: 8 workers summed, plus the serial tail. PyTorch's are from `torch.profiler`:

| CPU ms per step | rustlab 0.3.11 | PyTorch `float64`, 1 thread | PyTorch `float64`, 10 threads |
|---|---:|---:|---:|
| Matrix multiply | 145 | 12.2 | 11.9 |
| GELU and its gradient | 21 | 26.9 (scalar `tanh`) | 4.4 |
| Everything else: copies, allocation, indexing, softmax, LayerNorm, optimizer | about 170 | about 17 | about 14 |
| Wall time | 53–55 | 53–56 | 28–30 |

Elementwise math is already level: `gelu` on 1032 × 512 takes 0.57 ms in rustlab and 0.53 ms in PyTorch on 10 threads. The gap is matrix multiply (12×), then everything around the arithmetic (about 10×).

## What changed in the picture

The step is two parts:

| Part of the 53.0 ms step (run 08) | ms | Share | What limits it |
|---|---:|---:|---|
| 8 windows in `parmap` (`par.grad_flat_B8`) | 41.4 | 78 % | Matrix multiply on vector units, and each window costing 1.68× its serial time when 8 run at once |
| Serial optimizer tail: unflatten, reduce, clip, AdamW, sampling | 11.6 | 22 % | Full passes over 811 k numbers, and a serial `sum` |

The worker profile (`perf/profile_step.rlab`, 8 windows, CPU summed over threads, 322 ms) shows where the 1.68× goes. Matrix multiply is 145 ms of it (45 %). The rest is memory traffic that eight threads compete for:

| Worker profile row | Calls | Self ms | Bytes produced | Serial equivalent |
|---|---:|---:|---:|---|
| `parmap:[matmul]` | 1,176 | 145–148 | | 14.5 ms a window serially |
| `parmap:zeros` | 248 | 23.5–24.9 | 300 MB | 95–100 µs a call, against 24 µs serially |
| `parmap:outer` (library broadcasts) | 904 | 7.7–8.0 | 257 MB | See the optional item |
| `parmap:reshape` | 1,016 | 11.5–14.4 | 103 MB | |
| `parmap:gpt_flatten_tail` | 8 | 13.1 | 101 MB | 0.43 ms serially (`flat.flatten_811k`), 1.6 ms a window in a worker |
| `parmap:[index]`, `[index-assign]` | 11,816 | 29.7 | | |

The allocation keys isolate one part of it. `prim.alloc_window` runs one window's 20 `zeros` calls, about 37 MB as complex, and touches each array once. It takes 0.29 ms serially. Run as 8 `parmap` tasks (`par.alloc_window_B8`), it takes 1.48 ms, against 0.16 ms for 8 trivial tasks. So allocation alone runs 4–5× slower when 8 workers allocate at once. Allocating and filling the arrays shows the same thing: 1.04 ms serially, 2.7 ms as 8 tasks.

**0.3.16.** Releases 0.3.12–0.3.16 made no performance changes aimed at this workload. Run 09 recorded 0.3.16 about 10 % slower on every key, matrix multiply included. That run was taken while macOS's `syspolicyd` was using about 1.4 cores, and the outcome below records the side-by-side check against a 0.3.11 build.

## Blocking bug, fix first: `run` with two `..` levels fails since 0.3.14

Not a performance request, but it stops the training workload from running at all.

**Symptom.** On 0.3.14–0.3.16, `make sherlock-quick` fails at its first script:

```text
error: line 15: runtime error: run: ../../lib/gpt.rlab: No such file or directory (os error 2)
```

The same happens to 45 of this repo's 106 lesson scripts (`make lesson-02`), and to every `tutorial/sherlock/*.rlab`. 0.3.11 runs them all. The book still renders, because the notebook renderer sets a path jail, which skips the directory change.

**Cause.** In `exec_run_path` (`crates/rustlab-script/src/eval/line_io.rs`, 0.3.14's "`run` follows the script directory"), the steps are:
1. Change into `safe.parent()`.
2. Read `safe`, still relative, from the new directory.

So `run "../../lib/x.rlab"` changes into `../../lib`, then opens `../../lib/x.rlab` relative to that. A single `..` works only by coincidence (`lib/../lib`), which is why `perf/gate.rlab` (`run "../lib/gpt.rlab"`) still passes. Reproduced in a scratch tree, with the run statement in a script at `rt/a/b/two.rlab`:

| `run` path | 0.3.16 |
|---|---|
| `"y.rlab"` (same folder) | works |
| `"../lib/x.rlab"` (from `rt/a/one.rlab`) | works |
| `"../../lib/x.rlab"` | fails |
| `"a/b/../../lib/x.rlab"`, which never leaves the folder | fails |
| an absolute path | works |

**Fix.** Make the path absolute, for example `std::env::current_dir()?.join(&safe)`, before changing directory, or read the source before the change. Add a test with `../../` and with `a/b/../../`.

**Acceptance.** `make sherlock-quick` and `make lesson-02` run on the new build.

## Summary

| # | Request | Attacks | Keys and targets |
|---|---|---|---|
| R1 | `float64` matrix multiply on the SME matrix unit | 145 of 322 worker CPU ms; 12× PyTorch | `prim.matmul_512x512x512` ≥ 400 GFLOP/s; `prim.matmul_129x128x512` ≥ 200; `prim.matmul_A_Bt_129x128x128` ≥ 150 |
| R2 | Real-valued storage, first slice (planned in `real_storage.md`) | Halves every byte the workers move; the base for an `f32` mode | `parmap` out 101,376 → ≈ 50,688 KB; `par.grad_flat_B8` ≤ 1.4 × `model.grad_flat` |
| R3 | Allocation that does not slow down under `parmap` | 4–5× slower `zeros` in workers | `par.alloc_window_B8` ≤ 0.6 ms; `parmap:zeros` self ≤ 8 ms per 8 windows |
| R4 | The serial optimizer tail | 22 % of the step, a share that grows as the windows get faster | `flat.sumsq_811k` ≤ 0.4 ms; `flat.clip_811k` ≤ 0.6 ms; `flat.adamw_811k` ≤ 1.5 ms; `flat.unflatten_811k` ≤ 0.8 ms |

**Estimates.**
- **R1 alone:** worker CPU drops from 322 ms to about 190, `parmap` from 41 to about 24 ms, and the step to about 36 ms: roughly 28,000 tok/s.
- **R1–R4:** `parmap` about 17 ms, tail about 6 ms, a step of about 23 ms: roughly 44,000 tok/s, past PyTorch `float64`.

Success is set at 26,000 tok/s with the gate passing. Matching PyTorch's `float32` CPU numbers (55,000–62,000) would need an `f32` mode on top of R2.

## R1: `float64` matrix multiply on the SME matrix unit

**Evidence.** Matrix multiply is 45 % of worker CPU: 1,176 products per 8 windows, 145 ms. rustlab's `matrixmultiply` dgemm runs at 42–56 GFLOP/s at every model shape. Accelerate, through PyTorch, runs the same shapes at 327–814 GFLOP/s on one thread, using the matrix unit. This CPU has it (`FEAT_SME`, `FEAT_SME2`, `FEAT_SME_F64F64`, streaming vector length 512 bits, so a `float64` ZA tile is 8 × 8). PyTorch's whole step spends 12 ms in matrix multiply, against rustlab's 145.

**Problem.** The products run on general vector units at a fraction of what the hardware offers for exactly this operation.

**Suggested approach.** rustlab declined Accelerate in iteration 01 because its rules forbid FFI in core numerics. SME does not need FFI. `core::arch::asm!` is stable on aarch64, and a kernel can do all of the following inside one `asm!` block:
1. Enter streaming mode (`smstart`).
2. Accumulate `fmopa za<n>.d` outer products over packed 8-wide panels of A and B.
3. Store the tiles and leave (`smstop`).

Every SME instruction stays inside the asm block, because Rust code must not run in streaming mode. Notes for the design:
- **Detect at run time.** Check `sysctl hw.optional.arm.FEAT_SME_F64F64`, or `std::arch::is_aarch64_feature_detected!` where it covers SME. Keep `matrixmultiply` as the fallback everywhere else.
- **Size threshold.** Entering and leaving streaming mode has a fixed cost, so small products may stay on the existing path. The model's smallest products are 129 × 32 × 129 (attention scores per head), and its largest are 129 × 512 × 128 and 512 × 129 × 512.
- **Transposed operands.** These come from 0.3.11's transposed views: `A' * B` and `A * B'` arrive as strided views and are packed anyway.
- **Rounding.** The order of the sums changes, so the last digits change. The gate allows that. The trajectory move should be reported, and a move above 1e-10 relative explained.
- **Accelerate behind a feature flag** is the alternative with the least engineering. It would be off by default and outside core numerics, and would reach PyTorch's rates immediately, but it is FFI. Whether to allow it is a decision for rustlab's rules and its maintainer.

**Acceptance.** `prim.matmul_512x512x512` ≥ 400 GFLOP/s. `prim.matmul_129x128x512` ≥ 200. `prim.matmul_A_Bt_129x128x128` ≥ 150, and `prim.attn_scores_129x32x129` faster than today. `parmap:[matmul]` in `rustlab run --profile perf/profile_step.rlab` ≤ 30 ms per 8 windows. Gate PASS.

## R2: real-valued storage, the first slice

**Evidence.** The workers produce about 760 MB per 8 windows: `zeros` 300 MB, `outer` 257 MB, `reshape` 103 MB, flatten 101 MB. Half of every byte is an imaginary zero. Iteration 02's R1 cut copies on the serial side but left the parallel ratio at 1.7–1.8×. rustlab's own measurement ruled out the allocator alone as the cause, since `mimalloc` left `par.grad_flat_B8` unchanged, and pointed at shared cache and memory bandwidth. The bytes are the lever. Real storage is also what an `f32` mode would build on, and `f32` is worth another 1.5–1.7× in PyTorch.

**Request.** The first slice of `../rustlab/dev/plans/real_storage.md`, steps 0–6, as planned there: constructors, elementwise, `matmul_conj`, indexing, `reshape` / transpose / one-row `[…]`, scalar-function builtins, `sum` / `mean` / `max`, `parmap` packing, `save`. Then whatever `[widen]` still shows in `rustlab run --profile=ops perf/profile_step.rlab`. With R1 in place, a real-stored operand also skips the `real_parts` copy into the multiply.

**Acceptance.** The `parmap` row of `rustlab run --profile perf/profile_step.rlab` reports ≈ 50,688 KB out, and `parmap:zeros` ≈ 150 MB. No `[widen]` row above 1 ms in the profile. `par.grad_flat_B8` ≤ 1.4 × `model.grad_flat`. Gate PASS; `real_storage.md` aims at a trajectory bit-identical to 0.3.11, and the gate's 1e-6 limit is the hard bound.

## R3: allocation that does not slow down under `parmap`

**Evidence.** `par.alloc_window_B8` is 1.48 ms against 0.29 ms for `prim.alloc_window` run once, with 0.16 ms of `parmap` overhead. In the profile, `zeros` averages 95–100 µs a call in workers, against 24 µs serially. rustlab's iteration 02 note measured a local `mimalloc` build at 20,570–20,740 tok/s, 18 % faster than the same source with the system allocator. The gain came from the serial part of the step, and rustlab left it as a dependency decision under its Rule 10.

**Problem.** Each window allocates about 37 MB of fresh, zero-filled memory and frees it, on every step and in every worker. With the system allocator, allocations this large go to the operating system each time, and 8 threads doing so at once serialize.

**Suggested approach.** Either take the allocator decision, with its trade-off study, or keep large freed buffers for reuse per thread. Real storage (R2) halves the sizes either way. rustlab decides what fits its rules on dependencies.

**Acceptance.** `par.alloc_window_B8` ≤ 0.6 ms. `parmap:zeros` self time ≤ 8 ms per 8 windows in `rustlab run --profile perf/profile_step.rlab`. Gate PASS, trajectory unchanged.

## R4: the serial optimizer tail

**Evidence.** Between the `parmap` calls, each step runs in serial: unflatten 1.6 ms, gradient reduce 2.4 ms, clip 1.5 ms, AdamW 2.7 ms, and 3.5 ms of sampling and bookkeeping. That is 11.6 ms, 22 % of the step today. Once R1 shrinks the parallel part it becomes a third. PyTorch's whole optimizer step (clip and AdamW over 811 k `float64` values) is a few milliseconds. Measured on an 811,008-element vector:

| Operation | ms | Note |
|---|---:|---|
| `g .^ 2` | 0.21 | Parallel elementwise |
| `sum(g)` | 1.12 | 725 Melem/s: the reduction is the bottleneck |
| `sum(g .^ 2)` (`flat.sumsq_811k`, used by `gpt_clip`) | 1.42–1.54 | |
| `dot(g, g)` | 0.85 | |
| `norm(g) ^ 2` | 0.44 | Already fast, but rounds differently from `sum(g .^ 2)` |
| `sum(G8, 1)` on 8 × 811,009 | 0.70 | |
| `gpt_adamw` (`flat.adamw_811k`) | 2.68 | About 12 full-vector passes, each a fresh 13 MB temporary |
| `gpt_unflatten` (`flat.unflatten_811k`) | 1.59 | 14 contiguous slices `v(i:i+n-1)`, each copied, then reshaped |

**Problem.** `sum` over a long contiguous vector runs serially at under 1 Gelem/s, so the clip spends most of its time there. AdamW makes one pass, and one temporary, per operator.

**Suggested approach.** A vectorized reduction for `sum` over contiguous data, parallel above a size threshold, with fixed chunks so the result is deterministic from run to run. It will round differently from today's sequential sum, which the gate allows. For AdamW, fewer passes: either evaluating a whole elementwise expression in one pass, or reusing a uniquely owned operand's buffer for the result, which copy-on-write already detects. A contiguous range slice of a vector could share its parent's buffer the way `reshape` now does. rustlab chooses.

**Acceptance.** `flat.sumsq_811k` ≤ 0.4 ms, `flat.clip_811k` ≤ 0.6 ms, `flat.adamw_811k` ≤ 1.5 ms, `flat.unflatten_811k` ≤ 0.8 ms. Gate PASS. A new summation order moves the trajectory by round-off, which is expected. A move above 1e-10 relative should be explained in the response note.

## Optional, if convenient alongside R2: implicit expansion as fast as the same-shape operation

Not ranked: it is worth about 3 % of worker CPU (`outer` 8 ms and its 257 MB per 8 windows).

**Evidence.** rustlab expands a `1 × n` row down a `T × n` matrix, and a `T × 1` column across it. The library still broadcasts through `outer` (`gpt_rows`, `gpt_cols`). Run 05 switched the library to implicit expansion and had to revert it, because expansion is slower than building the full matrix and then operating on it:

| 129 × 128 | ms |
|---|---:|
| `X .* F`, with `F` already expanded (`prim.ewise_same_129x128`) | 0.014 |
| `outer` then `.*`, what the library does now | 0.025 |
| `X .* row` (`prim.expand_row_129x128`) | 0.028 |
| `X - col` (`prim.expand_col_129x128`) | 0.025 |

With expansion in place, `prim.layernorm_lib_129x128` went from 0.110 to 0.163 ms, and the step did not improve.

**Suggested approach.** A kernel for each expansion shape that walks the full operand contiguously and reads the row or column operand by index. No expanded temporary, and no per-element index arithmetic in the general broadcast path. The result must equal the same-shape operation on the expanded matrix bit for bit, which it does in 0.3.11.

**Acceptance.** `prim.expand_row_129x128` ≤ 0.014 ms and `prim.expand_col_129x128` ≤ 0.014 ms, that is no slower than `prim.ewise_same_129x128`. Then this repo switches the library over.

## This repo's items for run 03b

These are not requests to rustlab. They are done after rustlab's changes land, so the two gains stay separate.

| Item | Evidence | Change |
|---|---|---|
| Gradient reduce without the slice | `sum(Gm(:, 1:n), 1)` copies the 8 × 811 k slice (104 MB) before summing: 2.4 ms, against 0.70 ms for `sum(Gm, 1)` | In `train.rlab`, the other training scripts and `e2e.train_step_B8`: `sum(Gm, 1)`, then drop the loss column. Bit-identical. `flat.batch_reduce_8x811k` keeps its expression so the key keeps its meaning. |
| Broadcast by implicit expansion | 904 `outer` calls and 257 MB per 8 windows; run 05 is the change, ready to reapply | Only once the optional expansion item is met |
| A batched step: 8 windows stacked into one 1032-row forward and backward | PyTorch's shape. With R1 the matrix unit is shared, so 8 workers queueing small products on it may lose to one stream of large ones. It also removes `parmap` packing and the per-window flatten (101 MB). | Prototype `gpt_grad_batch` after R1 lands, and compare it with `par.grad_flat_B8`. The per-head attention loop over 8 windows needs a batched multiply (see Deferred). |
| Unflatten | 1.6 ms a step | Revisit after R2 and R4 |

## Deferred

| Item | Why it waits |
|---|---|
| `f32` mode | Worth 1.5–1.7× in PyTorch on this CPU. Needs real storage (R2) first, and its own gate tolerance. The next step after PyTorch `float64` parity. |
| Batched multiply (`pagemtimes`) on `Tensor3` | Needed by the batched step above. Rank it after R1, from the prototype. |
| Fused causal attention with backward | Attention is about 10 % of a window. PyTorch's fused attention gains 15 % in `float32`. |
| Threaded dgemm for large products | Superseded by R1: the matrix unit does not scale with threads, as PyTorch shows. |
| Parallel math builtins inside `parmap` workers | rustlab found them neutral there, because the cores are already busy. |

## Outcome (measured 2026-11-12, rustlab 0.3.64)

Measured as run 10, [`perf/reports/10-v0.3.64.md`](../reports/10-v0.3.64.md), on rustlab 0.3.64 (source `1c45aa6`), after this repo merged the 0.3.43–0.3.63 compatibility branch. rustlab's response notes for 0.3.17–0.3.64 have not been read here, so the statuses come only from the keys.

| # | Status | Measured (run 10) | Note |
|---|---|---|---|
| R1 | nearly met | `prim.matmul_512x512x512` 384 GFLOP/s (target 400); `prim.matmul_129x128x512` 231 (200); `prim.matmul_A_Bt_129x128x128` 134 (150) | 5–7× over run 08 at every matmul shape. |
| R2 | storage met, ratio not met | `parmap` out 50,688 KB (target ≈ 50,688); `par.grad_flat_B8` = 2.39 × `model.grad_flat` (1.4) | The ratio got worse because the serial window got 2.7× faster while packing and the shared matrix unit did not keep pace. |
| R3 | not met | `par.alloc_window_B8` 0.84 ms (target 0.6) | 1.85× faster than run 08. |
| R4 | partly met | `flat.sumsq_811k` 0.25 ms (0.4), `flat.clip_811k` 0.29 (0.6) met; `flat.adamw_811k` 1.94 (1.5), `flat.unflatten_811k` 1.23 (0.8) not | The tail is now about 5.5 ms of a 28.4 ms step. |

**Training step:** 19,310 → 36,050 tok/s (28.4 ms per step), 1.95× over run 08 and 16.05× the 0.3.9 baseline. That is past the 26,000 success line and the 35,000 goal (PyTorch `float64`, 34,300–37,200). By the stopping rule the flywheel pauses here. A new goal would be PyTorch `float32` (55,000–62,000), which needs an `f32` mode.

**Real tutorial:** `make sherlock` took 2 min 3 s for every stage. Pre-training took 88 s at 34,871 tok/s and reached the same 2.203 held-out bits/char. SFT scored 40 / 46 (42 on 0.3.11), the known last-digit sensitivity. `make sherlock-quick` took 15 s.

**Run 03b library items:** the batched step is done (below, not adopted). The gradient-reduce and unflatten items are still open.

### Run 03b: the batched step (2026-11-12, rustlab 0.3.64; not adopted)

`perf/batch_step.rlab` stacks the B windows into one (B T) × d forward and backward (`gpt_grad_batch`). Embedding, LayerNorm, the QKV / Wo / FFN / LM-head products and the loss run once over all B T rows. Attention loops over (window, head). The embedding gradient is one one-hot product. `perf/batch_check.rlab` compares it with the `parmap` path on the bench model (d 128, 4 layers, 4 heads, T 129, B 8).

**Correct:** the mean gradient matches `parmap` to 2.4e-15 of its largest entry, and the loss to 8.9e-16, both unmasked and with a random SFT-style mask.

**Slower at every grouping** (median of 7, same machine and binary):

| Layout | ms | tok/s |
|---|---:|---:|
| `parmap` × 8 windows (`gpt_grad_flat`, today) | 21.4 | 47,800 |
| `parmap` × 8 groups of 1 (`gpt_grad_batch`) | 26.7 | 38,400 |
| `parmap` × 4 groups of 2 | 33.9 | 30,200 |
| `parmap` × 2 groups of 4 | 54.0 | 18,900 |
| one stream of 8 | 99.3 | 10,300 |

**Why** (`rustlab run --profile`, one stream, 95 ms a call):
- 768 of the 820 products per call are the 129 × 32 × 129 attention products, at about 40 µs each: about 30 ms of overhead, not arithmetic. In the `parmap` path, 8 workers share them.
- LayerNorm, softmax, GELU and the broadcasts take about 35 ms. Per element they cost the same on 1,032 rows as on 129, because rustlab runs them on one core at any size.
- The stacked products got taller as intended, but they were only a few ms of the step.

So on 0.3.64, stacking windows turns parallel work into serial work, and the matrix unit is not the limit. A batched step would need two rustlab features: a batched multiply (`pagemtimes`, Deferred) for the per-(window, head) attention products, and threaded element-wise operations on large arrays. Rank them in iteration 04 against this prototype. `lib/gpt.rlab` is unchanged.
