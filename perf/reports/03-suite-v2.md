# Perf loop run 03 — suite-v2

- Date: 2026-10-18T20:02:18
- rustlab: rustlab 0.3.10 (binary 2026-10-18T19:56:48; source 1ebb762: Perf flywheel iteration 01: real matmul, nested profiler, parmap and elementwise copies (0.3.10) (#77))
- Machine: Apple M4 Max, 14 cores
- Note: same binary as run 02 (rustlab 0.3.10); bench suite v2 adds prim.zeros_512x384, prim.row_assign_loop_129x128, prim.matmul_A_Bt_129x128x128, prim.matmul_At_B_512x129x512, flat.reshape_811k, prim.call_struct_arg_13MB

**Correctness gate: PASS** — gradient check 7.79e-08 (limit 1e-06); trajectory matches reference (worst rel diff 0, limit 1e-06)

## Headline

| workload | now | vs previous | vs baseline |
|---|---:|---:|---:|
| training step, B = 8 | 9,542 tok/s | 0.96× | 4.25× |
| parmap gradient, B = 8 | 1.244e+04 tok/s | 0.98× | 3.91× |
| one window forward + backward (serial) | 2,918 tok/s | 0.99× | 3.47× |
| matmul 512^3 | 55.99 GFLOP/s | 1.02× | 6.30× |
| matmul at the FFN shape | 46.49 GFLOP/s | 0.97× | 5.27× |

## Where a training step goes

| part | ms | % of step |
|---|---:|---:|
| model forward + backward, 8 windows (spread over the cores) | 82.3 | 77 % |
| unflatten theta -> struct | 4.7 | 4 % |
| reduce 8 gradient rows | 2.4 | 2 % |
| clip | 2.1 | 2 % |
| AdamW on flat vectors | 4.4 | 4 % |
| other (sampling windows, interpreter) | 11.4 | 11 % |
| parmap efficiency vs 8 x serial | 53 % | |

## Every workload

| key | ms | rate | vs previous | vs baseline |
|---|---:|---:|---:|---:|
| `e2e.train_step_B8` | 107.311 | 9,542 tok/s | 0.96× | 4.25× |
| `flat.adamw_811k` | 4.392 |  | 1.03× | 4.44× |
| `flat.axpy_811k` | 0.262 | 3,100 Melem/s | 1.27× | 6.78× |
| `flat.batch_reduce_8x811k` | 2.408 |  | 0.97× | 12.19× |
| `flat.clip_811k` | 2.121 |  | 1.08× | 2.12× |
| `flat.flatten_811k` | 3.978 |  | 0.99× | 2.39× |
| `flat.reshape_811k` | 4.004 |  | — | — |
| `flat.sumsq_811k` | 1.471 | 551.5 Melem/s | 0.97× | 1.73× |
| `flat.unflatten_811k` | 4.676 |  | 0.99× | 4.38× |
| `model.backward` | 22.078 | 5,798 tok/s | 0.98× | 4.16× |
| `model.forward` | 14.804 | 8,647 tok/s | 0.99× | 3.11× |
| `model.generate_20_greedy` | 80.510 | 248.4 tok/s | 0.97× | 3.22× |
| `model.grad_flat` | 43.864 | 2,918 tok/s | 0.99× | 3.47× |
| `model.params` | 811008.000 |  | 1.00× | 1.00× |
| `par.grad_flat_B1` | 45.734 | 2,799 tok/s | 0.99× | 4.09× |
| `par.grad_flat_B14` | 119.690 | 1.497e+04 tok/s | 1.04× | 5.27× |
| `par.grad_flat_B8` | 82.338 | 1.244e+04 tok/s | 0.98× | 3.91× |
| `prim.attn_scores_129x32x129` | 0.030 | 36 GFLOP/s | 1.01× | 3.77× |
| `prim.broadcast_rows_129x128` | 0.013 |  | 1.00× | 1.36× |
| `prim.call_struct_arg_13MB` | 0.215 |  | — | — |
| `prim.ewise_129x512` | 0.111 |  | 0.97× | 3.62× |
| `prim.gelu_builtin_129x512` | 0.346 |  | 0.95× | 0.95× |
| `prim.gelu_grad_lib_129x512` | 0.393 |  | 1.00× | 6.95× |
| `prim.layernorm_lib_129x128` | 0.165 |  | 0.99× | 1.91× |
| `prim.matmul_129x128x384` | 0.270 | 46.98 GFLOP/s | 0.98× | 5.26× |
| `prim.matmul_129x128x512` | 0.364 | 46.49 GFLOP/s | 0.97× | 5.27× |
| `prim.matmul_129x128x77` | 0.066 | 38.6 GFLOP/s | 1.00× | 4.50× |
| `prim.matmul_129x512x128` | 0.379 | 44.67 GFLOP/s | 0.97× | 5.22× |
| `prim.matmul_512x512x512` | 4.794 | 55.99 GFLOP/s | 1.02× | 6.30× |
| `prim.matmul_A_Bt_129x128x128` | 0.113 | 37.55 GFLOP/s | — | — |
| `prim.matmul_At_B_512x129x512` | 1.265 | 53.47 GFLOP/s | — | — |
| `prim.matmul_transA_512x129x128` | 0.364 | 46.42 GFLOP/s | 0.99× | 5.24× |
| `prim.region_write_128x384` | 0.092 |  | 0.94× | 0.95× |
| `prim.row_assign_loop_129x128` | 0.120 |  | — | — |
| `prim.scalar_loop_1e5` | 13.483 | 7.417 Miter/s | 0.96× | 1.15× |
| `prim.scatter_add_loop_129` | 0.229 |  | 1.00× | 4.79× |
| `prim.slice_cols_128x32` | 0.002 |  | 1.14× | 9.02× |
| `prim.slice_rows_128x384` | 0.014 |  | 0.99× | 7.29× |
| `prim.softmax_129x129` | 0.058 |  | 0.99× | 1.01× |
| `prim.tanh_129x512` | 0.248 |  | 0.95× | 1.00× |
| `prim.transpose_129x512` | 0.026 |  | 1.00× | 1.83× |
| `prim.zeros_512x384` | 0.023 |  | — | — |

## `rustlab run --profile profile_step.rlab`

Sorted by self time. `Total` is inclusive. `parmap:` rows are worker CPU time summed over threads.

| function | calls | total ms | self ms | in KB | out KB |
|---|---:|---:|---:|---:|---:|
| `parmap:[matmul]` | 1176 | 149.6 | 149.6 | 0 | 0 |
| `parmap:gpt_backward` | 8 | 240.6 | 93.5 | 1,258 | 0 |
| `parmap:gpt_grad_flat` | 8 | 563.9 | 87.9 | 32 | 101,376 |
| `parmap` | 1 | 82.4 | 82.4 | 0 | 101,376 |
| `parmap:gpt_forward` | 8 | 177.9 | 68.6 | 32 | 1,242 |
| `[matmul]` | 441 | 48.5 | 48.5 | 0 | 0 |
| `parmap:gpt_flatten` | 8 | 53.4 | 35.9 | 0 | 101,376 |
| `parmap:zeros` | 248 | 28.5 | 28.5 | 4 | 299,885 |
| `gpt_backward` | 3 | 66.2 | 20.9 | 472 | 0 |
| `parmap:reshape` | 1008 | 17.7 | 17.7 | 103,383 | 103,368 |
| `parmap:gelu_grad` | 32 | 17.5 | 17.5 | 33,024 | 33,024 |
| `parmap:gelu` | 32 | 16.1 | 16.1 | 33,024 | 33,024 |
| `reshape` | 428 | 14.5 | 14.5 | 89,446 | 89,439 |
| `gpt_forward` | 3 | 44.4 | 14.3 | 12 | 466 |
| `parmap:gpt_ln_bwd` | 72 | 15.7 | 10.9 | 37,441 | 18,864 |
| `gpt_grad_flat` | 3 | 130.9 | 10.5 | 12 | 38,016 |
| `parmap:gpt_ln_fwd` | 72 | 13.9 | 9.6 | 18,864 | 37,297 |
| `parmap:softmax` | 144 | 7.6 | 7.6 | 35,746 | 35,746 |
| `[top-level]` | 1 | 245.2 | 7.3 | 0 | 0 |
| `gpt_flatten` | 4 | 12.7 | 7.0 | 0 | 50,688 |
