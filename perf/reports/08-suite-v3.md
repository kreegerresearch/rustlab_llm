# Perf loop run 08 — suite-v3

- Date: 2026-10-20T07:19:52
- rustlab: rustlab 0.3.11 (binary 2026-10-20T07:10:29; source 09cf101: Close out perf flywheel iteration 02: plan to closed/, revision 78b6c49 (#80))
- Machine: Apple M4 Max, 14 cores
- Note: Same binary and library as run 07; suite v3 adds prim.ewise_same/expand_row/expand_col_129x128, prim.alloc_window, par.alloc_window_B8 for iteration 03.

**Correctness gate: PASS** — gradient check 7.93e-08 (limit 1e-06); trajectory matches reference (worst rel diff 2.03e-16, limit 1e-06)

## Headline

| workload | now | vs previous | vs baseline |
|---|---:|---:|---:|
| training step, B = 8 | 1.931e+04 tok/s | 1.04× | 8.60× |
| parmap gradient, B = 8 | 2.474e+04 tok/s | 1.03× | 7.79× |
| one window forward + backward (serial) | 5,180 tok/s | 1.00× | 6.16× |
| matmul 512^3 | 57.81 GFLOP/s | 1.02× | 6.50× |
| matmul at the FFN shape | 52.2 GFLOP/s | 1.09× | 5.92× |

## Where a training step goes

| part | ms | % of step |
|---|---:|---:|
| model forward + backward, 8 windows (spread over the cores) | 41.4 | 78 % |
| unflatten theta -> struct | 1.6 | 3 % |
| reduce 8 gradient rows | 2.4 | 4 % |
| clip | 1.5 | 3 % |
| AdamW on flat vectors | 2.7 | 5 % |
| other (sampling windows, interpreter) | 3.5 | 7 % |
| parmap efficiency vs 8 x serial | 60 % | |

## Every workload

| key | ms | rate | vs previous | vs baseline |
|---|---:|---:|---:|---:|
| `e2e.train_step_B8` | 53.037 | 1.931e+04 tok/s | 1.04× | 8.60× |
| `flat.adamw_811k` | 2.675 |  | 1.04× | 7.29× |
| `flat.axpy_811k` | 0.284 | 2,858 Melem/s | 0.89× ⚠ | 6.25× |
| `flat.batch_reduce_8x811k` | 2.378 |  | 1.06× | 12.34× |
| `flat.clip_811k` | 1.522 |  | 1.15× | 2.95× |
| `flat.flatten_811k` | 0.434 |  | 1.11× | 21.91× |
| `flat.reshape_811k` | 0.142 |  | 0.95× | — |
| `flat.sumsq_811k` | 1.419 | 571.3 Melem/s | 1.05× | 1.80× |
| `flat.unflatten_811k` | 1.594 |  | 1.00× | 12.84× |
| `model.backward` | 14.195 | 9,017 tok/s | 1.00× | 6.48× |
| `model.forward` | 9.080 | 1.41e+04 tok/s | 1.02× | 5.06× |
| `model.generate_20_greedy` | 48.180 | 415.1 tok/s | 1.00× | 5.38× |
| `model.grad_flat` | 24.708 | 5,180 tok/s | 1.00× | 6.16× |
| `model.params` | 811008.000 |  | 1.00× | 1.00× |
| `par.alloc_window_B8` | 1.483 |  | — | — |
| `par.grad_flat_B1` | 25.438 | 5,032 tok/s | 1.01× | 7.36× |
| `par.grad_flat_B14` | 80.281 | 2.232e+04 tok/s | 0.80× ⚠ | 7.85× |
| `par.grad_flat_B8` | 41.389 | 2.474e+04 tok/s | 1.03× | 7.79× |
| `prim.alloc_window` | 0.294 |  | — | — |
| `prim.attn_scores_129x32x129` | 0.028 | 37.65 GFLOP/s | 1.00× | 3.94× |
| `prim.broadcast_rows_129x128` | 0.007 |  | 0.99× | 2.55× |
| `prim.call_struct_arg_13MB` | 0.002 |  | 0.94× | — |
| `prim.ewise_129x512` | 0.101 |  | 1.00× | 4.01× |
| `prim.ewise_same_129x128` | 0.014 |  | — | — |
| `prim.expand_col_129x128` | 0.025 |  | — | — |
| `prim.expand_row_129x128` | 0.028 |  | — | — |
| `prim.gelu_builtin_129x512` | 0.160 |  | 1.00× | 2.05× |
| `prim.gelu_grad_lib_129x512` | 0.145 |  | 1.07× | 18.88× |
| `prim.layernorm_lib_129x128` | 0.110 |  | 0.99× | 2.87× |
| `prim.matmul_129x128x384` | 0.237 | 53.6 GFLOP/s | 1.04× | 6.00× |
| `prim.matmul_129x128x512` | 0.324 | 52.2 GFLOP/s | 1.09× | 5.92× |
| `prim.matmul_129x128x77` | 0.055 | 46.09 GFLOP/s | 1.02× | 5.37× |
| `prim.matmul_129x512x128` | 0.331 | 51.16 GFLOP/s | 1.05× | 5.98× |
| `prim.matmul_512x512x512` | 4.644 | 57.81 GFLOP/s | 1.02× | 6.50× |
| `prim.matmul_A_Bt_129x128x128` | 0.089 | 47.25 GFLOP/s | 1.11× | — |
| `prim.matmul_At_B_512x129x512` | 1.196 | 56.55 GFLOP/s | 1.04× | — |
| `prim.matmul_transA_512x129x128` | 0.307 | 55.08 GFLOP/s | 1.05× | 6.21× |
| `prim.region_write_128x384` | 0.009 |  | 1.00× | 9.71× |
| `prim.row_assign_loop_129x128` | 0.082 |  | 1.07× | — |
| `prim.scalar_loop_1e5` | 7.781 | 12.85 Miter/s | 0.99× | 1.99× |
| `prim.scatter_add_loop_129` | 0.174 |  | 0.98× | 6.29× |
| `prim.slice_cols_128x32` | 0.002 |  | 0.93× | 7.09× |
| `prim.slice_rows_128x384` | 0.013 |  | 0.99× | 8.25× |
| `prim.softmax_129x129` | 0.048 |  | 0.99× | 1.22× |
| `prim.tanh_129x512` | 0.105 |  | 1.16× | 2.36× |
| `prim.transpose_129x512` | 0.023 |  | 0.99× | 2.08× |
| `prim.zeros_512x384` | 0.023 |  | 1.00× | — |

⚠ More than 10 % slower than the previous run: `flat.axpy_811k`, `par.grad_flat_B14`

## `rustlab run --profile profile_step.rlab`

Sorted by self time. `Total` is inclusive. `parmap:` rows are worker CPU time summed over threads.

| function | calls | total ms | self ms | in KB | out KB |
|---|---:|---:|---:|---:|---:|
| `parmap:[matmul]` | 1176 | 148.1 | 148.1 | 0 | 0 |
| `parmap` | 1 | 47.4 | 47.4 | 0 | 101,376 |
| `[matmul]` | 441 | 43.4 | 43.4 | 0 | 0 |
| `parmap:gpt_backward` | 8 | 165.1 | 29.7 | 1,258 | 0 |
| `parmap:zeros` | 248 | 23.5 | 23.5 | 4 | 299,885 |
| `parmap:gpt_forward` | 8 | 117.2 | 21.3 | 32 | 1,242 |
| `parmap:gpt_flatten_tail` | 8 | 24.4 | 13.1 | 0 | 101,376 |
| `parmap:reshape` | 1016 | 11.5 | 11.5 | 103,400 | 103,384 |
| `parmap:gelu_grad` | 32 | 11.2 | 11.2 | 33,024 | 33,024 |
| `parmap:gelu` | 32 | 10.1 | 10.1 | 33,024 | 33,024 |
| `parmap:softmax` | 144 | 8.1 | 8.1 | 35,746 | 35,746 |
| `parmap:outer` | 904 | 8.0 | 8.0 | 3,818 | 257,426 |
| `parmap:gpt_ln_bwd` | 72 | 12.6 | 7.9 | 37,441 | 18,864 |
| `gpt_backward` | 3 | 43.9 | 6.8 | 472 | 0 |
| `randn` | 6 | 6.8 | 6.8 | 0 | 12,596 |
| `parmap:gpt_ln_fwd` | 72 | 11.2 | 6.6 | 18,864 | 37,297 |
| `parmap:sum` | 624 | 4.8 | 4.8 | 186,023 | 1,446 |
| `gpt_forward` | 3 | 28.9 | 4.6 | 12 | 466 |
| `reshape` | 431 | 4.6 | 4.6 | 89,452 | 89,445 |
| `gpt_flatten_tail` | 4 | 6.8 | 3.6 | 0 | 50,688 |
