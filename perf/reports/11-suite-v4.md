# Perf loop run 11 — suite-v4

- Date: 2026-11-12T17:20:32
- rustlab: rustlab 0.3.64 (binary 2026-11-12T12:39:01; source 1c45aa6: FDTD Phase 6: patch antenna capstone and parallel stepping (0.3.64))
- Machine: Apple M4 Max, 14 cores
- Note: same binary and library as run 10; suite v4 adds prim.softmax_129x129, prim.attn_head_129x32, prim.row_block_write/read_129x128, par.matmul_129x128x512_B8, par.gelu_129x512_B8 for iteration 04

**Correctness gate: PASS** — gradient check 7.93e-08 (limit 1e-06); trajectory matches reference (worst rel diff 1.42e-15, limit 1e-06)

## Headline

| workload | now | vs previous | vs baseline |
|---|---:|---:|---:|
| training step, B = 8 | 3.699e+04 tok/s | 1.03× | 16.47× |
| parmap gradient, B = 8 | 3.957e+04 tok/s | 0.96× | 12.45× |
| one window forward + backward (serial) | 1.24e+04 tok/s | 1.01× | 14.74× |
| matmul 512^3 | 380.2 GFLOP/s | 0.99× | 42.76× |
| matmul at the FFN shape | 231.1 GFLOP/s | 1.00× | 26.21× |

## Where a training step goes

| part | ms | % of step |
|---|---:|---:|
| model forward + backward, 8 windows (spread over the cores) | 25.9 | 93 % |
| unflatten theta -> struct | 1.2 | 4 % |
| reduce 8 gradient rows | 1.7 | 6 % |
| clip | 0.3 | 1 % |
| AdamW on flat vectors | 2.0 | 7 % |
| other (sampling windows, interpreter) | 0.0 | 0 % |
| parmap efficiency vs 8 x serial | 40 % | |

## Every workload

| key | ms | rate | vs previous | vs baseline |
|---|---:|---:|---:|---:|
| `e2e.train_step_B8` | 27.681 | 3.699e+04 tok/s | 1.03× | 16.47× |
| `flat.adamw_811k` | 2.036 |  | 0.95× | 9.58× |
| `flat.axpy_811k` | 0.330 | 2,461 Melem/s | 0.96× | 5.38× |
| `flat.batch_reduce_8x811k` | 1.723 |  | 1.03× | 17.04× |
| `flat.clip_811k` | 0.285 |  | 1.03× | 15.73× |
| `flat.flatten_811k` | 0.382 |  | 1.09× | 24.88× |
| `flat.reshape_811k` | 0.101 |  | 0.80× ⚠ | — |
| `flat.sumsq_811k` | 0.210 | 3,863 Melem/s | 1.18× | 12.15× |
| `flat.unflatten_811k` | 1.232 |  | 0.99× | 16.61× |
| `model.backward` | 5.297 | 2.416e+04 tok/s | 0.99× | 17.36× |
| `model.forward` | 4.553 | 2.811e+04 tok/s | 0.98× | 10.10× |
| `model.generate_20_greedy` | 27.913 | 716.5 tok/s | 1.00× | 9.29× |
| `model.grad_flat` | 10.321 | 1.24e+04 tok/s | 1.01× | 14.74× |
| `model.params` | 811008.000 |  | 1.00× | 1.00× |
| `par.alloc_window_B8` | 0.761 |  | 1.10× | — |
| `par.gelu_129x512_B8` | 7.225 | 1,463 Melem/s | — | — |
| `par.grad_flat_B1` | 11.080 | 1.155e+04 tok/s | 1.00× | 16.90× |
| `par.grad_flat_B14` | 35.929 | 4.988e+04 tok/s | 1.01× | 17.54× |
| `par.grad_flat_B8` | 25.878 | 3.957e+04 tok/s | 0.96× | 12.45× |
| `par.matmul_129x128x512_B8` | 4.479 | 604 GFLOP/s | — | — |
| `prim.alloc_window` | 0.108 |  | 1.01× | — |
| `prim.attn_head_129x32` | 0.092 |  | — | — |
| `prim.attn_scores_129x32x129` | 0.012 | 91.29 GFLOP/s | 0.93× | 9.55× |
| `prim.broadcast_rows_129x128` | 0.004 |  | 0.83× ⚠ | 4.28× |
| `prim.call_struct_arg_13MB` | 0.002 |  | 0.96× | — |
| `prim.ewise_129x512` | 0.026 |  | 1.00× | 15.40× |
| `prim.ewise_same_129x128` | 0.003 |  | 0.99× | — |
| `prim.expand_col_129x128` | 0.003 |  | 1.00× | — |
| `prim.expand_row_129x128` | 0.003 |  | 1.00× | — |
| `prim.gelu_builtin_129x512` | 0.100 |  | 0.97× | 3.27× |
| `prim.gelu_grad_lib_129x512` | 0.148 |  | 0.87× ⚠ | 18.40× |
| `prim.layernorm_lib_129x128` | 0.050 |  | 0.98× | 6.36× |
| `prim.matmul_129x128x384` | 0.054 | 235 GFLOP/s | 1.02× | 26.32× |
| `prim.matmul_129x128x512` | 0.073 | 231.1 GFLOP/s | 1.00× | 26.21× |
| `prim.matmul_129x128x77` | 0.019 | 132.7 GFLOP/s | 1.00× | 15.47× |
| `prim.matmul_129x512x128` | 0.078 | 215.6 GFLOP/s | 1.00× | 25.21× |
| `prim.matmul_512x512x512` | 0.706 | 380.2 GFLOP/s | 0.99× | 42.76× |
| `prim.matmul_A_Bt_129x128x128` | 0.032 | 133.7 GFLOP/s | 0.99× | — |
| `prim.matmul_At_B_512x129x512` | 0.203 | 333.8 GFLOP/s | 1.01× | — |
| `prim.matmul_transA_512x129x128` | 0.061 | 276.2 GFLOP/s | 1.00× | 31.17× |
| `prim.region_write_128x384` | 0.006 |  | 0.99× | 14.27× |
| `prim.row_assign_loop_129x128` | 0.077 |  | 0.98× | — |
| `prim.row_block_read_129x128` | 0.002 |  | — | — |
| `prim.row_block_write_129x128` | 0.002 |  | — | — |
| `prim.scalar_loop_1e5` | 7.323 | 13.66 Miter/s | 1.01× | 2.11× |
| `prim.scatter_add_loop_129` | 0.166 |  | 0.98× | 6.59× |
| `prim.slice_cols_128x32` | 0.002 |  | 0.98× | 9.45× |
| `prim.slice_rows_128x384` | 0.005 |  | 0.99× | 20.83× |
| `prim.softmax_129x129` | 0.036 | 457.2 Melem/s | 0.99× | 1.60× |
| `prim.tanh_129x512` | 0.083 |  | 1.02× | 2.99× |
| `prim.transpose_129x512` | 0.018 |  | 1.00× | 2.73× |
| `prim.zeros_512x384` | 0.007 |  | 1.02× | — |

⚠ More than 10 % slower than the previous run: `flat.reshape_811k`, `prim.broadcast_rows_129x128`, `prim.gelu_grad_lib_129x512`

## `rustlab run --profile profile_step.rlab`

Sorted by self time. `Total` is inclusive. `parmap:` rows are worker CPU time summed over threads.

| function | calls | total ms | self ms | in KB | out KB |
|---|---:|---:|---:|---:|---:|
| `parmap:[matmul]` | 1176 | 50.4 | 50.4 | 334,526 | 167,263 |
| `parmap` | 1 | 24.4 | 24.4 | 0 | 50,688 |
| `parmap:gpt_backward` | 8 | 70.4 | 20.4 | 629 | 0 |
| `parmap:gpt_forward` | 8 | 62.9 | 18.7 | 16 | 621 |
| `[matmul]` | 441 | 11.7 | 11.7 | 125,447 | 62,724 |
| `parmap:gelu` | 32 | 9.3 | 9.3 | 16,512 | 16,512 |
| `parmap:gelu_grad` | 32 | 9.0 | 9.0 | 16,512 | 16,512 |
| `parmap:reshape` | 1016 | 8.5 | 8.5 | 51,708 | 51,692 |
| `parmap:softmax` | 144 | 8.0 | 8.0 | 34,514 | 17,873 |
| `randn` | 6 | 6.3 | 6.3 | 0 | 6,298 |
| `parmap:gpt_flatten_tail` | 8 | 14.2 | 5.9 | 0 | 50,688 |
| `gpt_backward` | 3 | 17.0 | 4.2 | 236 | 0 |
| `gpt_forward` | 3 | 14.4 | 3.6 | 6 | 233 |
| `parmap:outer` | 904 | 3.5 | 3.5 | 1,909 | 128,713 |
| `parmap:gpt_ln_fwd` | 72 | 5.8 | 3.2 | 9,432 | 18,649 |
| `reshape` | 431 | 3.1 | 3.1 | 44,729 | 44,722 |
| `parmap:gpt_ln_bwd` | 72 | 6.2 | 3.0 | 18,721 | 9,432 |
| `parmap:sum` | 624 | 2.7 | 2.7 | 93,014 | 723 |
| `gelu_grad` | 12 | 2.5 | 2.5 | 6,192 | 6,192 |
| `softmax` | 54 | 2.4 | 2.4 | 12,943 | 6,702 |
