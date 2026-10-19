# Perf loop run 00 — baseline

- Date: 2026-10-18T17:45:06
- rustlab: rustlab 0.3.9 (binary 2026-10-18T15:58:45; source 65d6507: Ship HTML libraries beside the page instead of from a CDN. (#76))
- Machine: Apple M4 Max, 14 cores
- Note: rustlab 0.3.9 as released: complex-f64 storage, matrixmultiply zgemm, no BLAS

**Correctness gate: PASS** — gradient check 7.79e-08 (limit 1e-06); no reference yet: this run's trajectory is now perf/reference/gate.json

## Headline

| workload | now | vs previous | vs baseline |
|---|---:|---:|---:|
| training step, B = 8 | 2,246 tok/s | — | — |
| parmap gradient, B = 8 | 3,177 tok/s | — | — |
| one window forward + backward (serial) | 841.4 tok/s | — | — |
| matmul 512^3 | 8.89 GFLOP/s | — | — |
| matmul at the FFN shape | 8.817 GFLOP/s | — | — |

## Where a training step goes

| part | ms | % of step |
|---|---:|---:|
| model forward + backward, 8 windows (spread over the cores) | 322.3 | 71 % |
| unflatten theta -> struct | 20.5 | 4 % |
| reduce 8 gradient rows | 29.4 | 6 % |
| clip | 4.5 | 1 % |
| AdamW on flat vectors | 19.5 | 4 % |
| other (sampling windows, interpreter) | 59.8 | 13 % |
| parmap efficiency vs 8 x serial | 47 % | |

## Every workload

| key | ms | rate | vs previous | vs baseline |
|---|---:|---:|---:|---:|
| `e2e.train_step_B8` | 455.903 | 2,246 tok/s | — | — |
| `flat.adamw_811k` | 19.500 |  | — | — |
| `flat.axpy_811k` | 1.774 | 457.2 Melem/s | — | — |
| `flat.batch_reduce_8x811k` | 29.355 |  | — | — |
| `flat.clip_811k` | 4.486 |  | — | — |
| `flat.flatten_811k` | 9.499 |  | — | — |
| `flat.sumsq_811k` | 2.550 | 318.1 Melem/s | — | — |
| `flat.unflatten_811k` | 20.467 |  | — | — |
| `model.backward` | 91.948 | 1,392 tok/s | — | — |
| `model.forward` | 45.980 | 2,784 tok/s | — | — |
| `model.generate_20_greedy` | 259.220 | 77.15 tok/s | — | — |
| `model.grad_flat` | 152.128 | 841.4 tok/s | — | — |
| `model.params` | 811008.000 |  | — | — |
| `par.grad_flat_B1` | 187.250 | 683.6 tok/s | — | — |
| `par.grad_flat_B14` | 630.209 | 2,844 tok/s | — | — |
| `par.grad_flat_B8` | 322.308 | 3,177 tok/s | — | — |
| `prim.attn_scores_129x32x129` | 0.111 | 9.555 GFLOP/s | — | — |
| `prim.broadcast_rows_129x128` | 0.017 |  | — | — |
| `prim.ewise_129x512` | 0.403 |  | — | — |
| `prim.gelu_builtin_129x512` | 0.327 |  | — | — |
| `prim.gelu_grad_lib_129x512` | 2.732 |  | — | — |
| `prim.layernorm_lib_129x128` | 0.315 |  | — | — |
| `prim.matmul_129x128x384` | 1.420 | 8.928 GFLOP/s | — | — |
| `prim.matmul_129x128x512` | 1.918 | 8.817 GFLOP/s | — | — |
| `prim.matmul_129x128x77` | 0.296 | 8.577 GFLOP/s | — | — |
| `prim.matmul_129x512x128` | 1.977 | 8.552 GFLOP/s | — | — |
| `prim.matmul_512x512x512` | 30.194 | 8.89 GFLOP/s | — | — |
| `prim.matmul_transA_512x129x128` | 1.908 | 8.862 GFLOP/s | — | — |
| `prim.region_write_128x384` | 0.087 |  | — | — |
| `prim.scalar_loop_1e5` | 15.464 | 6.467 Miter/s | — | — |
| `prim.scatter_add_loop_129` | 1.096 |  | — | — |
| `prim.slice_cols_128x32` | 0.017 |  | — | — |
| `prim.slice_rows_128x384` | 0.104 |  | — | — |
| `prim.softmax_129x129` | 0.058 |  | — | — |
| `prim.tanh_129x512` | 0.249 |  | — | — |
| `prim.transpose_129x512` | 0.048 |  | — | — |

## `rustlab run --profile profile_step.rlab`

| function | calls | total ms | in KB | out KB |
|---|---:|---:|---:|---:|
| `gpt_grad_flat` | 3 | 468.3 | 12 | 38,016 |
| `parmap` | 1 | 257.3 | 0 | 101,376 |
| `gpt_unflatten` | 3 | 63.5 | 38,016 | 0 |
| `gpt_init` | 1 | 20.4 | 0 | 0 |
| `gpt_flatten` | 1 | 11.7 | 0 | 12,672 |
| `rand` | 4 | 0.0 | 0 | 22 |
| `gpt_config` | 1 | 0.0 | 0 | 0 |
| `floor` | 4 | 0.0 | 22 | 22 |
| `seed` | 1 | 0.0 | 0 | 0 |
| `ones` | 1 | 0.0 | 0 | 2 |
