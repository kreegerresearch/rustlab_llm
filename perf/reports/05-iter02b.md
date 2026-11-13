# Perf loop run 05 — iter02b

- Date: 2026-10-20T07:16:22
- rustlab: rustlab 0.3.11 (binary 2026-10-20T07:10:29; source 09cf101: Close out perf flywheel iteration 02: plan to closed/, revision 78b6c49 (#80))
- Machine: Apple M4 Max, 14 cores
- Note: lib/gpt.rlab: implicit expansion (gpt_row/gpt_col) instead of outer-product broadcasts; gpt_dlogits in one linear-index write. Bit-identical gate output.

**Correctness gate: PASS** — gradient check 7.93e-08 (limit 1e-06); trajectory matches reference (worst rel diff 2.03e-16, limit 1e-06)

## Headline

| workload | now | vs previous | vs baseline |
|---|---:|---:|---:|
| training step, B = 8 | 1.755e+04 tok/s | 1.00× | 7.81× |
| parmap gradient, B = 8 | 2.201e+04 tok/s | 0.96× | 6.93× |
| one window forward + backward (serial) | 4,821 tok/s | 0.92× | 5.73× |
| matmul 512^3 | 55.42 GFLOP/s | 0.96× | 6.23× |
| matmul at the FFN shape | 47.65 GFLOP/s | 0.92× | 5.40× |

## Where a training step goes

| part | ms | % of step |
|---|---:|---:|
| model forward + backward, 8 windows (spread over the cores) | 46.5 | 80 % |
| unflatten theta -> struct | 1.4 | 2 % |
| reduce 8 gradient rows | 2.4 | 4 % |
| clip | 1.6 | 3 % |
| AdamW on flat vectors | 2.7 | 5 % |
| other (sampling windows, interpreter) | 3.8 | 6 % |
| parmap efficiency vs 8 x serial | 57 % | |

## Every workload

| key | ms | rate | vs previous | vs baseline |
|---|---:|---:|---:|---:|
| `e2e.train_step_B8` | 58.345 | 1.755e+04 tok/s | 1.00× | 7.81× |
| `flat.adamw_811k` | 2.655 |  | 1.04× | 7.34× |
| `flat.axpy_811k` | 0.267 | 3,035 Melem/s | 1.01× | 6.64× |
| `flat.batch_reduce_8x811k` | 2.430 |  | 0.97× | 12.08× |
| `flat.clip_811k` | 1.603 |  | 1.12× | 2.80× |
| `flat.flatten_811k` | 0.444 |  | 0.86× ⚠ | 21.37× |
| `flat.reshape_811k` | 0.133 |  | 0.98× | — |
| `flat.sumsq_811k` | 1.400 | 579.2 Melem/s | 1.07× | 1.82× |
| `flat.unflatten_811k` | 1.358 |  | 1.07× | 15.07× |
| `model.backward` | 15.681 | 8,163 tok/s | 0.92× | 5.86× |
| `model.forward` | 10.161 | 1.26e+04 tok/s | 0.90× ⚠ | 4.53× |
| `model.generate_20_greedy` | 53.591 | 373.2 tok/s | 0.89× ⚠ | 4.84× |
| `model.grad_flat` | 26.551 | 4,821 tok/s | 0.92× | 5.73× |
| `model.params` | 811008.000 |  | 1.00× | 1.00× |
| `par.grad_flat_B1` | 27.697 | 4,621 tok/s | 0.93× | 6.76× |
| `par.grad_flat_B14` | 67.054 | 2.672e+04 tok/s | 1.00× | 9.40× |
| `par.grad_flat_B8` | 46.527 | 2.201e+04 tok/s | 0.96× | 6.93× |
| `prim.attn_scores_129x32x129` | 0.028 | 37.81 GFLOP/s | 0.99× | 3.96× |
| `prim.broadcast_rows_129x128` | 0.007 |  | 0.90× ⚠ | 2.31× |
| `prim.call_struct_arg_13MB` | 0.002 |  | 1.12× | — |
| `prim.ewise_129x512` | 0.112 |  | 0.89× ⚠ | 3.59× |
| `prim.gelu_builtin_129x512` | 0.147 |  | 1.14× | 2.22× |
| `prim.gelu_grad_lib_129x512` | 0.184 |  | 0.95× | 14.81× |
| `prim.layernorm_lib_129x128` | 0.163 |  | 0.67× ⚠ | 1.93× |
| `prim.matmul_129x128x384` | 0.260 | 48.82 GFLOP/s | 0.93× | 5.47× |
| `prim.matmul_129x128x512` | 0.355 | 47.65 GFLOP/s | 0.92× | 5.40× |
| `prim.matmul_129x128x77` | 0.061 | 41.38 GFLOP/s | 0.91× | 4.82× |
| `prim.matmul_129x512x128` | 0.360 | 47.01 GFLOP/s | 0.92× | 5.50× |
| `prim.matmul_512x512x512` | 4.843 | 55.42 GFLOP/s | 0.96× | 6.23× |
| `prim.matmul_A_Bt_129x128x128` | 0.100 | 42.45 GFLOP/s | 0.97× | — |
| `prim.matmul_At_B_512x129x512` | 1.261 | 53.65 GFLOP/s | 0.95× | — |
| `prim.matmul_transA_512x129x128` | 0.333 | 50.77 GFLOP/s | 0.91× | 5.73× |
| `prim.region_write_128x384` | 0.010 |  | 0.89× ⚠ | 8.70× |
| `prim.row_assign_loop_129x128` | 0.088 |  | 0.90× ⚠ | — |
| `prim.scalar_loop_1e5` | 8.220 | 12.17 Miter/s | 0.91× | 1.88× |
| `prim.scatter_add_loop_129` | 0.192 |  | 0.90× ⚠ | 5.72× |
| `prim.slice_cols_128x32` | 0.002 |  | 1.08× | 7.78× |
| `prim.slice_rows_128x384` | 0.014 |  | 0.90× ⚠ | 7.42× |
| `prim.softmax_129x129` | 0.050 |  | 0.95× | 1.17× |
| `prim.tanh_129x512` | 0.093 |  | 1.22× | 2.69× |
| `prim.transpose_129x512` | 0.023 |  | 0.99× | 2.09× |
| `prim.zeros_512x384` | 0.025 |  | 0.93× | — |

⚠ More than 10 % slower than the previous run: `flat.flatten_811k`, `model.forward`, `model.generate_20_greedy`, `prim.broadcast_rows_129x128`, `prim.ewise_129x512`, `prim.layernorm_lib_129x128`, `prim.region_write_128x384`, `prim.row_assign_loop_129x128`, `prim.scatter_add_loop_129`, `prim.slice_rows_128x384`

## `rustlab run --profile profile_step.rlab`

Sorted by self time. `Total` is inclusive. `parmap:` rows are worker CPU time summed over threads.

| function | calls | total ms | self ms | in KB | out KB |
|---|---:|---:|---:|---:|---:|
| `parmap:[matmul]` | 1176 | 146.6 | 146.6 | 0 | 0 |
| `parmap` | 1 | 52.5 | 52.5 | 0 | 101,376 |
| `[matmul]` | 441 | 46.3 | 46.3 | 0 | 0 |
| `parmap:gpt_backward` | 8 | 170.7 | 35.9 | 1,258 | 0 |
| `parmap:gpt_forward` | 8 | 125.1 | 26.1 | 32 | 1,242 |
| `parmap:zeros` | 248 | 22.7 | 22.7 | 4 | 299,885 |
| `parmap:reshape` | 520 | 16.3 | 16.3 | 102,392 | 102,384 |
| `parmap:gpt_ln_bwd` | 72 | 16.9 | 14.4 | 37,441 | 18,864 |
| `parmap:gpt_ln_fwd` | 72 | 15.6 | 13.9 | 18,864 | 37,297 |
| `parmap:gpt_grad_flat` | 8 | 335.5 | 11.7 | 32 | 101,376 |
| `parmap:gpt_flatten` | 8 | 27.2 | 10.9 | 0 | 101,376 |
| `parmap:gelu_grad` | 32 | 10.6 | 10.6 | 33,024 | 33,024 |
| `parmap:gelu` | 32 | 10.5 | 10.5 | 33,024 | 33,024 |
| `gpt_backward` | 3 | 49.6 | 9.3 | 472 | 0 |
| `parmap:softmax` | 144 | 8.2 | 8.2 | 35,746 | 35,746 |
| `randn` | 6 | 7.1 | 7.1 | 0 | 12,596 |
| `gpt_forward` | 3 | 32.3 | 6.0 | 12 | 466 |
| `gpt_ln_bwd` | 27 | 5.4 | 4.6 | 14,040 | 7,074 |
| `parmap:sum` | 624 | 4.6 | 4.6 | 186,023 | 1,446 |
| `reshape` | 245 | 4.2 | 4.2 | 89,074 | 89,070 |
