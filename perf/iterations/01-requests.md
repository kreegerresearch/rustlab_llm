# Perf flywheel iteration 01: requests to rustlab

- **Measured on:** [`perf/reports/00-baseline.md`](../reports/00-baseline.md), rustlab 0.3.9, source `65d6507`, Apple M4 Max with 14 cores, 2026-10-18.
- **Training step now:** 2,246 tok/s (`e2e.train_step_B8`, 456 ms per step); goal 17,000 tok/s.
- **Protocol:** [`docs/perf-flywheel.md`](../../docs/perf-flywheel.md). Check a development build without installing it:

  ```sh
  RUSTLAB=$PWD/target/release/rustlab make -C ../rustlab_llm perf-check
  ```

## Where a training step goes today

| Part of one B = 8 step | ms | Share |
|---|---:|---:|
| `parmap` of 8 forward + backward windows | 322 | 71 % |
| Per-step flat-vector work: unflatten 20, reduce 29, clip 4.5, AdamW 19.5 | 73 | 16 % |
| Other: mostly `parmap` setup, which scales with the caller's environment (see R3) | 60 | 13 % |

Inside one window, `model.grad_flat` takes 152 ms. About 65 % of it is matrix multiply, running at 8.6–8.9 GFLOP/s at every shape. The rest is elementwise work, slicing, and interpreter overhead. Eight windows on 14 cores take 322 ms, 2.1× one window, where perfect scaling would take 1×.

## Summary

| # | Request | Attacks | Expected step time after | Keys and targets |
|---|---|---|---:|---|
| R1 | Real-valued matrix multiply fast path | 65 % of every window | 456 → ≈ 290 ms | `prim.matmul_512x512x512` ≥ 40 GFLOP/s; `model.grad_flat` ≤ 80 ms |
| R2 | A profiler that sees inside user functions | Diagnosis for every later turn | no direct change | `profile_step.rlab` shows nested functions and matrix multiply |
| R3 | `parmap` without per-task environment clones | 47 % parallel efficiency; 13 % "other" | ≈ 290 → ≈ 200 ms | `par.grad_flat_B8` ≤ 1.3 × `model.grad_flat`; B14 tok/s ≥ B8 |
| R4 | Real fast paths for elementwise ops and reductions | 16 % flat-vector work, ≈ 15 % of each window | ≈ 200 → ≈ 130 ms | `flat.adamw_811k` ≤ 5 ms; `flat.batch_reduce_8x811k` ≤ 5 ms |

Landing all four should take the step from 2,246 to roughly 7,500 tok/s, about 3.3×. The default Sherlock run would drop from 20 minutes to about 6. The iteration is a success at 5,000 tok/s or more with the gate passing.

## R1: real-valued matrix multiply fast path

**Evidence.** Every multiply in the suite runs at 8.6–8.9 GFLOP/s, whatever its shape: `prim.matmul_129x128x512` takes 1.92 ms, and `prim.matmul_512x512x512` takes 30.2 ms. Values are `Array2<Complex<f64>>` (`rustlab-core/src/types.rs`: `CMatrix`). `*` is `a.dot(b)` at `rustlab-script/src/eval/value.rs:1920`, which ndarray sends to `matrixmultiply::zgemm`, single-threaded, with no BLAS linked. Every operand in the training workload is real. The complex multiply does four times the floating-point work and moves twice the bytes.

**Problem.** Real workloads pay for complex arithmetic in the operation that dominates their run time.

**Suggested approach.** When both operands have all-zero imaginary parts, multiply the real parts with a real GEMM and widen the result. The check is O(n²) against O(n³) work. Options are `matrixmultiply::dgemm`, already a dependency through ndarray, which needs nothing new, or Accelerate on macOS through ndarray's `blas` feature, with OpenBLAS elsewhere. A per-value "known real" flag set by constructors such as `randn`, `zeros` and real arithmetic would remove the scan later. A real storage variant is the full fix, but is larger, so it is deferred below.

**Acceptance.** `prim.matmul_512x512x512` ≥ 40 GFLOP/s, or ≥ 100 with Accelerate. `prim.matmul_129x128x512` ≥ 25 GFLOP/s. `model.grad_flat` ≤ 80 ms. Gate PASS: BLAS reorders sums, which the 1e-6 tolerance allows.

**Library follow-up.** None. `lib/gpt.rlab` benefits unchanged.

## R2: a profiler that sees inside user functions

**Evidence.** In `perf/reports/00-baseline.md`, the whole forward and backward pass is a single `gpt_grad_flat` row, 468 ms for 3 calls. That is by design today: `Profiler::enter_higher_order` in `rustlab-script/src/eval/profile.rs` suppresses recording while a user function runs, so only outermost calls appear. Operators such as `*`, `.*`, indexing and indexed assignment are not calls, so they never appear. Separately, the in-script `profile()` prints no report at exit, contrary to its help text. Only an explicit `profile_report()` prints. Repro:

```text
function y = inner(x)
  y = x * x';
end
profile()
A = randn(64, 64);
z = inner(A);
# expected: a report on stderr at exit; actual: nothing
```

**Problem.** Every hotspot in this file had to be found by hand-instrumented copies of library functions. The flywheel needs the profiler to do that.

**Suggested approach.** Record nested user-function calls, with inclusive and self time per function, or a call tree. Add pseudo-rows for matrix multiply and, behind an option such as `--profile=ops`, for elementwise operators and index reads and writes. Aggregate `parmap` worker time under the callee. Fix the missing exit report for `profile()`.

**Acceptance.** `rustlab run --profile perf/profile_step.rlab` lists `gpt_forward`, `gpt_backward`, `gpt_ln_fwd`, `gpt_gelu_grad`, and a matrix-multiply row. The self times of all rows sum to within 10 % of the script's run time. Profiling overhead is ≤ 10 % on `model.grad_flat`. In-script `profile()` prints its report at exit.

**Library follow-up.** `perf/run.py` will store the richer table, and later request files will cite it instead of hand-instrumented timings.

## R3: `parmap` without per-task environment clones

**Evidence.** Eight windows in `parmap` take 322 ms (`par.grad_flat_B8`) against 152 ms for one serial window (`model.grad_flat`). Perfect scaling on 14 cores would be about 152 ms, so efficiency is 47 %. Fourteen windows give fewer tokens per second than eight: 2,844 against 3,177. A trivial lambda over 8 tasks takes 0.14 ms in a small environment, but 6.4 ms when the caller's environment holds one 8 × 811 k matrix. The training step's environment holds several such values. The cause is at `rustlab-script/src/eval/parmap.rs:151–179`, where `par_iter` clones the evaluator and the callable for every element. The comment there assumes the clone costs well under 1 ms, which is false when the environment holds the model's parameters, gradients and optimizer state. The profile also shows `parmap` returning 101 MB for 8 gradient rows, 16 bytes per number.

**Problem.** Data-parallel training pays copies proportional to the caller's whole environment for every task, and loses half its cores to that and to allocation.

**Suggested approach.** Give the worker evaluator only the lambda's captured values, not the caller's environment. Share those values read-only across workers, for example with `Arc` and copy-on-write, rather than deep-copying them. Reuse one evaluator per worker thread through rayon's `map_with`. Write results directly into the output matrix.

**Acceptance.** `par.grad_flat_B8` ≤ 1.3 × `model.grad_flat`. `par.grad_flat_B14` tokens per second ≥ `par.grad_flat_B8`. A trivial 8-task `parmap` costs the same, within 2×, whatever the caller's environment size. Gate PASS.

**Library follow-up.** None. `tutorial/sherlock/train.rlab` benefits unchanged.

## R4: real fast paths for elementwise ops and reductions

**Evidence.** The per-step optimizer work on 811,008 parameters takes 73 ms. AdamW is 19.5 ms, about 10 elementwise passes. Reducing 8 gradient rows takes 29.4 ms. Unflattening takes 20.5 ms, which is only slicing and reshaping 13 MB. A plain `theta - 0.001 * g` runs at 457 M elements/s, where a real f64 loop on this machine typically reaches several G elements/s; that last figure is an estimate. Inside a window, the library's GELU derivative takes 2.73 ms (`prim.gelu_grad_lib_129x512`), against 0.33 ms for the builtin forward `gelu`. `H .* H + 0.5 * H` on a 129 × 512 matrix takes 0.40 ms.

**Problem.** Every elementwise operator and reduction does complex arithmetic and allocates a temporary per operator.

**Suggested approach.** Use real-only kernels when operands are real, the same check or flag as R1. Avoid a temporary per operator where an expression can be evaluated in one pass, for example `a*x + b*y`. Make column-range slicing and reshape of vectors copy at memory speed. Add a builtin for the GELU derivative, for example `gelu_grad(x)` or `gelu(x, "grad")`, matching the builtin `gelu`'s formula.

**Acceptance.** `flat.adamw_811k` ≤ 5 ms. `flat.batch_reduce_8x811k` ≤ 5 ms. `flat.unflatten_811k` ≤ 5 ms. `flat.axpy_811k` ≥ 1,500 M elements/s. If a GELU-derivative builtin lands, it runs at ≤ 0.5 ms on 129 × 512. Gate PASS.

**Library follow-up.** If the builtin lands, `gpt_gelu_grad` in `lib/gpt.rlab` calls it, measured separately as run `01b`.

## Deferred to later iterations

These were measured but are not yet worth asking for, or depend on R1–R4.

| Item | Current share | Why it waits |
|---|---|---|
| Real-valued storage for `Matrix` and `Vector` | Behind R1 and R4 | The full fix, but a large refactor. R1 and R4's fast paths show how much it is worth first. |
| Evaluator value-borrow refactor (rustlab roadmap C11) | Part of interpreter overhead | Its plan says it needs benchmarks first. This harness provides them. Re-rank after R2 shows interpreter self time. |
| Batched multiply (`pagemtimes`) for the per-head loop | ≈ 10 % of a window | Worth more once R1 makes the large multiplies cheap. |
| Fused causal attention with backward | ≈ 15 % of a window | Same reasoning, and a bigger API. |
| `f32` mode | Up to 2× more | Needs its own gate tolerance. Ask after the `f64` paths are fast. |
| Reverse-mode autodiff | Removes code, not time | A capability request, tracked in AGENTS.md. |

## Outcome (filled in after verification)

| # | Status | Measured | Note |
|---|---|---|---|
| R1 | open | | |
| R2 | open | | |
| R3 | open | | |
| R4 | open | | |
