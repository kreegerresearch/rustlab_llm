# Perf loop run 02 — iter01b

- Date: 2026-10-18T19:58:56
- rustlab: rustlab 0.3.10 (binary 2026-10-18T19:56:48; source 1ebb762: Perf flywheel iteration 01: real matmul, nested profiler, parmap and elementwise copies (0.3.10) (#77))
- Machine: Apple M4 Max, 14 cores
- Note: lib/gpt.rlab adopts 0.3.10: gelu_grad builtin; vectorised gpt_sinusoidal_pe (bit-identical, ~100x faster), which gpt_unflatten calls every step

**Correctness gate: PASS** — gradient check 7.79e-08 (limit 1e-06); trajectory matches reference (worst rel diff 0, limit 1e-06)

## Headline

| workload | now | vs previous | vs baseline |
|---|---:|---:|---:|
| training step, B = 8 | 9,967 tok/s | 1.13× | 4.44× |
| parmap gradient, B = 8 | 1.267e+04 tok/s | 1.06× | 3.99× |
| one window forward + backward (serial) | 2,940 tok/s | 1.07× | 3.49× |
| matmul 512^3 | 54.9 GFLOP/s | 0.98× | 6.18× |
| matmul at the FFN shape | 48.02 GFLOP/s | 1.04× | 5.45× |

## Where a training step goes

| part | ms | % of step |
|---|---:|---:|
| model forward + backward, 8 windows (spread over the cores) | 80.8 | 79 % |
| unflatten theta -> struct | 4.6 | 5 % |
| reduce 8 gradient rows | 2.3 | 2 % |
| clip | 2.3 | 2 % |
| AdamW on flat vectors | 4.5 | 4 % |
| other (sampling windows, interpreter) | 8.1 | 8 % |
| parmap efficiency vs 8 x serial | 54 % | |

## Every workload

| key | ms | rate | vs previous | vs baseline |
|---|---:|---:|---:|---:|
| `e2e.train_step_B8` | 102.735 | 9,967 tok/s | 1.13× | 4.44× |
| `flat.adamw_811k` | 4.508 |  | 1.01× | 4.33× |
| `flat.axpy_811k` | 0.333 | 2,432 Melem/s | 0.86× ⚠ | 5.32× |
| `flat.batch_reduce_8x811k` | 2.330 |  | 1.05× | 12.60× |
| `flat.clip_811k` | 2.297 |  | 0.91× | 1.95× |
| `flat.flatten_811k` | 3.924 |  | 1.05× | 2.42× |
| `flat.sumsq_811k` | 1.425 | 569.1 Melem/s | 0.98× | 1.79× |
| `flat.unflatten_811k` | 4.641 |  | 3.11× | 4.41× |
| `model.backward` | 21.690 | 5,901 tok/s | 1.12× | 4.24× |
| `model.forward` | 14.721 | 8,695 tok/s | 1.00× | 3.12× |
| `model.generate_20_greedy` | 78.408 | 255.1 tok/s | 1.01× | 3.31× |
| `model.grad_flat` | 43.544 | 2,940 tok/s | 1.07× | 3.49× |
| `model.params` | 811008.000 |  | 1.00× | 1.00× |
| `par.grad_flat_B1` | 45.336 | 2,823 tok/s | 1.06× | 4.13× |
| `par.grad_flat_B14` | 124.711 | 1.437e+04 tok/s | 1.03× | 5.05× |
| `par.grad_flat_B8` | 80.833 | 1.267e+04 tok/s | 1.06× | 3.99× |
| `prim.attn_scores_129x32x129` | 0.030 | 35.8 GFLOP/s | 1.00× | 3.75× |
| `prim.broadcast_rows_129x128` | 0.013 |  | 1.00× | 1.35× |
| `prim.ewise_129x512` | 0.108 |  | 1.03× | 3.72× |
| `prim.gelu_builtin_129x512` | 0.329 |  | 1.02× | 1.00× |
| `prim.gelu_grad_lib_129x512` | 0.391 |  | 2.25× | 6.98× |
| `prim.layernorm_lib_129x128` | 0.163 |  | 1.00× | 1.93× |
| `prim.matmul_129x128x384` | 0.264 | 47.96 GFLOP/s | 1.00× | 5.37× |
| `prim.matmul_129x128x512` | 0.352 | 48.02 GFLOP/s | 1.04× | 5.45× |
| `prim.matmul_129x128x77` | 0.066 | 38.67 GFLOP/s | 1.01× | 4.51× |
| `prim.matmul_129x512x128` | 0.369 | 45.85 GFLOP/s | 1.01× | 5.36× |
| `prim.matmul_512x512x512` | 4.890 | 54.9 GFLOP/s | 0.98× | 6.18× |
| `prim.matmul_transA_512x129x128` | 0.361 | 46.8 GFLOP/s | 1.00× | 5.28× |
| `prim.region_write_128x384` | 0.087 |  | 1.00× | 1.01× |
| `prim.scalar_loop_1e5` | 12.956 | 7.718 Miter/s | 1.02× | 1.19× |
| `prim.scatter_add_loop_129` | 0.230 |  | 1.00× | 4.77× |
| `prim.slice_cols_128x32` | 0.002 |  | 1.12× | 7.94× |
| `prim.slice_rows_128x384` | 0.014 |  | 1.01× | 7.40× |
| `prim.softmax_129x129` | 0.058 |  | 1.00× | 1.01× |
| `prim.tanh_129x512` | 0.237 |  | 1.01× | 1.05× |
| `prim.transpose_129x512` | 0.026 |  | 1.01× | 1.82× |

⚠ More than 10 % slower than the previous run: `flat.axpy_811k`

## `rustlab run --profile profile_step.rlab`

Sorted by self time. `Total` is inclusive. `parmap:` rows are worker CPU time summed over threads.

| function | calls | total ms | self ms | in KB | out KB |
|---|---:|---:|---:|---:|---:|
| `parmap:[matmul]` | 1176 | 150.1 | 150.1 | 0 | 0 |
| `parmap:gpt_backward` | 8 | 244.2 | 94.3 | 1,258 | 0 |
| `parmap:gpt_grad_flat` | 8 | 564.1 | 87.3 | 32 | 101,376 |
| `parmap` | 1 | 83.2 | 83.2 | 0 | 101,376 |
| `parmap:gpt_forward` | 8 | 175.3 | 68.2 | 32 | 1,242 |
| `[matmul]` | 441 | 48.9 | 48.9 | 0 | 0 |
| `parmap:gpt_flatten` | 8 | 53.5 | 35.1 | 0 | 101,376 |
| `parmap:zeros` | 248 | 28.3 | 28.3 | 4 | 299,885 |
| `gpt_backward` | 3 | 67.3 | 21.5 | 472 | 0 |
| `parmap:reshape` | 1008 | 18.6 | 18.6 | 103,383 | 103,368 |
| `parmap:gelu_grad` | 32 | 17.3 | 17.3 | 33,024 | 33,024 |
| `parmap:gelu` | 32 | 16.1 | 16.1 | 33,024 | 33,024 |
| `reshape` | 428 | 14.6 | 14.6 | 89,446 | 89,439 |
| `gpt_forward` | 3 | 44.6 | 14.5 | 12 | 466 |
| `parmap:gpt_ln_bwd` | 72 | 16.0 | 11.2 | 37,441 | 18,864 |
| `gpt_grad_flat` | 3 | 131.7 | 10.1 | 12 | 38,016 |
| `parmap:gpt_ln_fwd` | 72 | 14.1 | 9.8 | 18,864 | 37,297 |
| `parmap:softmax` | 144 | 8.2 | 8.2 | 35,746 | 35,746 |
| `[top-level]` | 1 | 246.6 | 7.3 | 0 | 0 |
| `gpt_flatten` | 4 | 12.5 | 6.9 | 0 | 50,688 |
