# Perf loop run 07 — iter02d

- Date: 2026-10-20T07:18:20
- rustlab: rustlab 0.3.11 (binary 2026-10-20T07:10:29; source 09cf101: Close out perf flywheel iteration 02: plan to closed/, revision 78b6c49 (#80))
- Machine: Apple M4 Max, 14 cores
- Note: lib/gpt.rlab: gpt_grad_flat appends the loss inside the flatten concatenation (gpt_flatten_tail) instead of a second 811k copy. Bit-identical gate output.

**Correctness gate: PASS** — gradient check 7.93e-08 (limit 1e-06); trajectory matches reference (worst rel diff 2.03e-16, limit 1e-06)

## Headline

| workload | now | vs previous | vs baseline |
|---|---:|---:|---:|
| training step, B = 8 | 1.864e+04 tok/s | 1.00× | 8.30× |
| parmap gradient, B = 8 | 2.4e+04 tok/s | 1.05× | 7.55× |
| one window forward + backward (serial) | 5,166 tok/s | 1.02× | 6.14× |
| matmul 512^3 | 56.45 GFLOP/s | 1.01× | 6.35× |
| matmul at the FFN shape | 48.01 GFLOP/s | 1.01× | 5.45× |

## Where a training step goes

| part | ms | % of step |
|---|---:|---:|
| model forward + backward, 8 windows (spread over the cores) | 42.7 | 78 % |
| unflatten theta -> struct | 1.6 | 3 % |
| reduce 8 gradient rows | 2.5 | 5 % |
| clip | 1.7 | 3 % |
| AdamW on flat vectors | 2.8 | 5 % |
| other (sampling windows, interpreter) | 3.6 | 7 % |
| parmap efficiency vs 8 x serial | 58 % | |

## Every workload

| key | ms | rate | vs previous | vs baseline |
|---|---:|---:|---:|---:|
| `e2e.train_step_B8` | 54.923 | 1.864e+04 tok/s | 1.00× | 8.30× |
| `flat.adamw_811k` | 2.772 |  | 1.01× | 7.03× |
| `flat.axpy_811k` | 0.253 | 3,201 Melem/s | 1.03× | 7.00× |
| `flat.batch_reduce_8x811k` | 2.518 |  | 0.94× | 11.66× |
| `flat.clip_811k` | 1.746 |  | 0.94× | 2.57× |
| `flat.flatten_811k` | 0.482 |  | 1.03× | 19.69× |
| `flat.reshape_811k` | 0.135 |  | 0.99× | — |
| `flat.sumsq_811k` | 1.488 | 545.2 Melem/s | 0.95× | 1.71× |
| `flat.unflatten_811k` | 1.595 |  | 0.95× | 12.83× |
| `model.backward` | 14.201 | 9,013 tok/s | 1.02× | 6.47× |
| `model.forward` | 9.267 | 1.381e+04 tok/s | 0.97× | 4.96× |
| `model.generate_20_greedy` | 48.361 | 413.6 tok/s | 1.02× | 5.36× |
| `model.grad_flat` | 24.779 | 5,166 tok/s | 1.02× | 6.14× |
| `model.params` | 811008.000 |  | 1.00× | 1.00× |
| `par.grad_flat_B1` | 25.582 | 5,004 tok/s | 1.01× | 7.32× |
| `par.grad_flat_B14` | 64.190 | 2.792e+04 tok/s | 1.01× | 9.82× |
| `par.grad_flat_B8` | 42.668 | 2.4e+04 tok/s | 1.05× | 7.55× |
| `prim.attn_scores_129x32x129` | 0.028 | 37.81 GFLOP/s | 1.00× | 3.96× |
| `prim.broadcast_rows_129x128` | 0.007 |  | 1.12× | 2.58× |
| `prim.call_struct_arg_13MB` | 0.002 |  | 0.99× | — |
| `prim.ewise_129x512` | 0.100 |  | 1.12× | 4.03× |
| `prim.gelu_builtin_129x512` | 0.159 |  | 0.78× ⚠ | 2.06× |
| `prim.gelu_grad_lib_129x512` | 0.155 |  | 1.16× | 17.67× |
| `prim.layernorm_lib_129x128` | 0.109 |  | 1.00× | 2.89× |
| `prim.matmul_129x128x384` | 0.247 | 51.42 GFLOP/s | 1.00× | 5.76× |
| `prim.matmul_129x128x512` | 0.352 | 48.01 GFLOP/s | 1.01× | 5.45× |
| `prim.matmul_129x128x77` | 0.056 | 45.31 GFLOP/s | 1.05× | 5.28× |
| `prim.matmul_129x512x128` | 0.345 | 48.94 GFLOP/s | 1.01× | 5.72× |
| `prim.matmul_512x512x512` | 4.755 | 56.45 GFLOP/s | 1.01× | 6.35× |
| `prim.matmul_A_Bt_129x128x128` | 0.100 | 42.47 GFLOP/s | 0.99× | — |
| `prim.matmul_At_B_512x129x512` | 1.240 | 54.55 GFLOP/s | 1.02× | — |
| `prim.matmul_transA_512x129x128` | 0.322 | 52.52 GFLOP/s | 1.01× | 5.93× |
| `prim.region_write_128x384` | 0.009 |  | 1.12× | 9.75× |
| `prim.row_assign_loop_129x128` | 0.087 |  | 0.90× ⚠ | — |
| `prim.scalar_loop_1e5` | 7.734 | 12.93 Miter/s | 1.05× | 2.00× |
| `prim.scatter_add_loop_129` | 0.171 |  | 1.11× | 6.40× |
| `prim.slice_cols_128x32` | 0.002 |  | 1.02× | 7.63× |
| `prim.slice_rows_128x384` | 0.013 |  | 1.12× | 8.31× |
| `prim.softmax_129x129` | 0.047 |  | 1.00× | 1.23× |
| `prim.tanh_129x512` | 0.122 |  | 0.93× | 2.04× |
| `prim.transpose_129x512` | 0.023 |  | 1.01× | 2.10× |
| `prim.zeros_512x384` | 0.023 |  | 1.00× | — |

⚠ More than 10 % slower than the previous run: `prim.gelu_builtin_129x512`, `prim.row_assign_loop_129x128`

## `rustlab run --profile profile_step.rlab`

Sorted by self time. `Total` is inclusive. `parmap:` rows are worker CPU time summed over threads.

| function | calls | total ms | self ms | in KB | out KB |
|---|---:|---:|---:|---:|---:|
| `parmap:[matmul]` | 1176 | 147.5 | 147.5 | 0 | 0 |
| `parmap` | 1 | 47.5 | 47.5 | 0 | 101,376 |
| `[matmul]` | 441 | 43.7 | 43.7 | 0 | 0 |
| `parmap:gpt_backward` | 8 | 161.5 | 29.2 | 1,258 | 0 |
| `parmap:zeros` | 248 | 23.1 | 23.1 | 4 | 299,885 |
| `parmap:gpt_forward` | 8 | 116.5 | 20.8 | 32 | 1,242 |
| `parmap:reshape` | 1016 | 12.9 | 12.9 | 103,400 | 103,384 |
| `parmap:gpt_flatten_tail` | 8 | 23.9 | 11.2 | 0 | 101,376 |
| `parmap:gelu_grad` | 32 | 9.9 | 9.9 | 33,024 | 33,024 |
| `parmap:gelu` | 32 | 9.2 | 9.2 | 33,024 | 33,024 |
| `parmap:softmax` | 144 | 8.5 | 8.5 | 35,746 | 35,746 |
| `parmap:gpt_ln_bwd` | 72 | 12.4 | 7.8 | 37,441 | 18,864 |
| `parmap:outer` | 904 | 7.7 | 7.7 | 3,818 | 257,426 |
| `gpt_backward` | 3 | 43.8 | 6.9 | 472 | 0 |
| `randn` | 6 | 6.8 | 6.8 | 0 | 12,596 |
| `parmap:gpt_ln_fwd` | 72 | 11.1 | 6.6 | 18,864 | 37,297 |
| `gpt_forward` | 3 | 29.2 | 4.7 | 12 | 466 |
| `parmap:sum` | 624 | 4.5 | 4.5 | 186,023 | 1,446 |
| `reshape` | 431 | 4.3 | 4.3 | 89,452 | 89,445 |
| `gpt_unflatten` | 3 | 4.9 | 3.4 | 38,016 | 0 |
