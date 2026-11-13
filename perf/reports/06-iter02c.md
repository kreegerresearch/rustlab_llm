# Perf loop run 06 — iter02c

- Date: 2026-10-20T07:17:23
- rustlab: rustlab 0.3.11 (binary 2026-10-20T07:10:29; source 09cf101: Close out perf flywheel iteration 02: plan to closed/, revision 78b6c49 (#80))
- Machine: Apple M4 Max, 14 cores
- Note: Revert run 05's implicit expansion (slower than outer in 0.3.11: layernorm_lib 0.110 -> 0.163 ms). Keep only gpt_dlogits in one linear-index write. Bit-identical gate output.

**Correctness gate: PASS** — gradient check 7.93e-08 (limit 1e-06); trajectory matches reference (worst rel diff 2.03e-16, limit 1e-06)

## Headline

| workload | now | vs previous | vs baseline |
|---|---:|---:|---:|
| training step, B = 8 | 1.87e+04 tok/s | 1.07× | 8.33× |
| parmap gradient, B = 8 | 2.292e+04 tok/s | 1.04× | 7.21× |
| one window forward + backward (serial) | 5,080 tok/s | 1.05× | 6.04× |
| matmul 512^3 | 56.14 GFLOP/s | 1.01× | 6.31× |
| matmul at the FFN shape | 47.7 GFLOP/s | 1.00× | 5.41× |

## Where a training step goes

| part | ms | % of step |
|---|---:|---:|
| model forward + backward, 8 windows (spread over the cores) | 44.7 | 82 % |
| unflatten theta -> struct | 1.5 | 3 % |
| reduce 8 gradient rows | 2.4 | 4 % |
| clip | 1.6 | 3 % |
| AdamW on flat vectors | 2.8 | 5 % |
| other (sampling windows, interpreter) | 1.8 | 3 % |
| parmap efficiency vs 8 x serial | 56 % | |

## Every workload

| key | ms | rate | vs previous | vs baseline |
|---|---:|---:|---:|---:|
| `e2e.train_step_B8` | 54.758 | 1.87e+04 tok/s | 1.07× | 8.33× |
| `flat.adamw_811k` | 2.798 |  | 0.95× | 6.97× |
| `flat.axpy_811k` | 0.261 | 3,109 Melem/s | 1.02× | 6.80× |
| `flat.batch_reduce_8x811k` | 2.365 |  | 1.03× | 12.41× |
| `flat.clip_811k` | 1.636 |  | 0.98× | 2.74× |
| `flat.flatten_811k` | 0.497 |  | 0.89× ⚠ | 19.11× |
| `flat.reshape_811k` | 0.134 |  | 0.99× | — |
| `flat.sumsq_811k` | 1.415 | 573.1 Melem/s | 0.99× | 1.80× |
| `flat.unflatten_811k` | 1.517 |  | 0.90× ⚠ | 13.49× |
| `model.backward` | 14.478 | 8,841 tok/s | 1.08× | 6.35× |
| `model.forward` | 9.035 | 1.417e+04 tok/s | 1.12× | 5.09× |
| `model.generate_20_greedy` | 49.321 | 405.5 tok/s | 1.09× | 5.26× |
| `model.grad_flat` | 25.199 | 5,080 tok/s | 1.05× | 6.04× |
| `model.params` | 811008.000 |  | 1.00× | 1.00× |
| `par.grad_flat_B1` | 25.722 | 4,976 tok/s | 1.08× | 7.28× |
| `par.grad_flat_B14` | 65.036 | 2.755e+04 tok/s | 1.03× | 9.69× |
| `par.grad_flat_B8` | 44.681 | 2.292e+04 tok/s | 1.04× | 7.21× |
| `prim.attn_scores_129x32x129` | 0.028 | 37.92 GFLOP/s | 1.00× | 3.97× |
| `prim.broadcast_rows_129x128` | 0.007 |  | 1.00× | 2.31× |
| `prim.call_struct_arg_13MB` | 0.002 |  | 0.92× | — |
| `prim.ewise_129x512` | 0.113 |  | 1.00× | 3.58× |
| `prim.gelu_builtin_129x512` | 0.125 |  | 1.18× | 2.63× |
| `prim.gelu_grad_lib_129x512` | 0.180 |  | 1.02× | 15.17× |
| `prim.layernorm_lib_129x128` | 0.109 |  | 1.50× | 2.88× |
| `prim.matmul_129x128x384` | 0.246 | 51.49 GFLOP/s | 1.05× | 5.77× |
| `prim.matmul_129x128x512` | 0.354 | 47.7 GFLOP/s | 1.00× | 5.41× |
| `prim.matmul_129x128x77` | 0.059 | 43.04 GFLOP/s | 1.04× | 5.02× |
| `prim.matmul_129x512x128` | 0.349 | 48.39 GFLOP/s | 1.03× | 5.66× |
| `prim.matmul_512x512x512` | 4.782 | 56.14 GFLOP/s | 1.01× | 6.31× |
| `prim.matmul_A_Bt_129x128x128` | 0.099 | 42.73 GFLOP/s | 1.01× | — |
| `prim.matmul_At_B_512x129x512` | 1.271 | 53.23 GFLOP/s | 0.99× | — |
| `prim.matmul_transA_512x129x128` | 0.324 | 52.16 GFLOP/s | 1.03× | 5.89× |
| `prim.region_write_128x384` | 0.010 |  | 1.00× | 8.70× |
| `prim.row_assign_loop_129x128` | 0.079 |  | 1.12× | — |
| `prim.scalar_loop_1e5` | 8.156 | 12.26 Miter/s | 1.01× | 1.90× |
| `prim.scatter_add_loop_129` | 0.190 |  | 1.01× | 5.76× |
| `prim.slice_cols_128x32` | 0.002 |  | 0.96× | 7.49× |
| `prim.slice_rows_128x384` | 0.014 |  | 1.00× | 7.42× |
| `prim.softmax_129x129` | 0.048 |  | 1.05× | 1.23× |
| `prim.tanh_129x512` | 0.113 |  | 0.82× ⚠ | 2.20× |
| `prim.transpose_129x512` | 0.023 |  | 0.99× | 2.08× |
| `prim.zeros_512x384` | 0.023 |  | 1.07× | — |

⚠ More than 10 % slower than the previous run: `flat.flatten_811k`, `flat.unflatten_811k`, `prim.tanh_129x512`

## `rustlab run --profile profile_step.rlab`

Sorted by self time. `Total` is inclusive. `parmap:` rows are worker CPU time summed over threads.

| function | calls | total ms | self ms | in KB | out KB |
|---|---:|---:|---:|---:|---:|
| `parmap:[matmul]` | 1176 | 145.5 | 145.5 | 0 | 0 |
| `parmap` | 1 | 50.6 | 50.6 | 0 | 101,376 |
| `[matmul]` | 441 | 44.5 | 44.5 | 0 | 0 |
| `parmap:gpt_backward` | 8 | 161.2 | 29.8 | 1,258 | 0 |
| `parmap:zeros` | 248 | 23.9 | 23.9 | 4 | 299,885 |
| `parmap:reshape` | 1016 | 22.0 | 22.0 | 103,400 | 103,384 |
| `parmap:gpt_forward` | 8 | 118.7 | 21.6 | 32 | 1,242 |
| `parmap:gpt_grad_flat` | 8 | 325.9 | 13.7 | 32 | 101,376 |
| `parmap:gelu` | 32 | 10.7 | 10.7 | 33,024 | 33,024 |
| `parmap:gpt_flatten` | 8 | 31.7 | 9.9 | 0 | 101,376 |
| `parmap:gelu_grad` | 32 | 9.7 | 9.7 | 33,024 | 33,024 |
| `parmap:outer` | 904 | 8.2 | 8.2 | 3,818 | 257,426 |
| `parmap:softmax` | 144 | 8.1 | 8.1 | 35,746 | 35,746 |
| `parmap:gpt_ln_bwd` | 72 | 12.7 | 7.9 | 37,441 | 18,864 |
| `gpt_backward` | 3 | 44.9 | 7.2 | 472 | 0 |
| `randn` | 6 | 7.0 | 7.0 | 0 | 12,596 |
| `parmap:gpt_ln_fwd` | 72 | 11.8 | 6.6 | 18,864 | 37,297 |
| `gpt_forward` | 3 | 29.4 | 4.7 | 12 | 466 |
| `parmap:sum` | 624 | 4.5 | 4.5 | 186,023 | 1,446 |
| `reshape` | 431 | 4.4 | 4.4 | 89,452 | 89,445 |
