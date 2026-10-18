# Sherlock tutorial: performance study and requests to rustlab

**Measured:** 2026-10-18, rustlab 0.3.9 (`~/.local/bin/rustlab`, no BLAS linked — `otool -L` shows none), Apple M4 Max, 14 logical CPUs, 38 GB, macOS 26.6.
**Script:** `tutorial/sherlock/benchmark.rlab` (`cd tutorial/sherlock && rustlab run benchmark.rlab --plot none`; add `--profile` for the function table). Every number below is pasted from one run of that script; the figure is `tutorial/sherlock/benchmark.svg`. Load note: TBD.
**Library under test:** `lib/gpt.rlab` — dense f64 matrix algebra, hand-written backward pass, one flat parameter vector, data parallelism over the batch with `parmap`.

## 1. What was measured

**Model-size sweep** — one window of T tokens (T+1 ids), forward / backward / flatten timed separately, means of 4 passes; `parmap(8)` is eight windows through `gpt_grad_flat` in one call. GFLOP/s uses the 6·params·T convention over forward + backward wall time. The 5 s early-stop guard never fired (the largest config is ≈ 1 s per window).

| d | L | H | d_ff | T | params | fwd ms | bwd ms | flat ms | serial tok/s | parmap(8) tok/s | GFLOP/s (6NT) |
|---|---|---|------|---|--------|--------|--------|---------|--------------|-----------------|---------------|
| 64 | 2 | 4 | 256 | 64 | 109440 | 4.9 | 9.5 | 1.29 | 4083 | 16398 | 2.92 |
| 128 | 4 | 4 | 512 | 128 | 811008 | 48.6 | 98.4 | 10.46 | 813 | 4122 | 4.24 |
| 192 | 6 | 6 | 768 | 128 | 2694528 | 144.4 | 287.3 | 36.93 | 273 | 1351 | 4.79 |
| 256 | 8 | 8 | 1024 | 128 | 6349824 | 329.2 | 643.6 | 89.94 | 120 | 456 | 5.01 |

**Steady state of the real run** (`train_run.log`, default model, B = 8, T = 128, including evaluation and checkpoint writes): 2,078–2,895 (median 2,629) tok/s per 50-step log line, 2,527 tok/s average over the whole run.

**parmap scaling** — default model, best of 3 steps per B, speedup against the serial window time from the sweep:

| B | s per step | tok/s | speedup | efficiency |
|---|-----------|-------|---------|------------|
| 1 | 0.178 | 718 | 0.88 | 88 % |
| 2 | 0.189 | 1354 | 1.66 | 83 % |
| 4 | 0.241 | 2123 | 2.61 | 65 % |
| 8 | 0.366 | 2797 | 3.44 | 43 % |
| 14 | 0.566 | 3164 | 3.89 | 28 % |

**Memory** — rustlab stores every number as `Complex<f64>` (16 bytes), so the resident model is twice the f64 figure; the NPZ checkpoint writes real arrays as float64 (`ckpt_latest.npz` is 19.47 MB for theta + m + v, matching 3 × 8 × 811 008):

| d | L | params | f64 weights MB | checkpoint MB (theta + m + v) | in-memory complex f64 MB | activation cache MB (T window) |
|---|---|--------|----------------|-------------------------------|--------------------------|--------------------------------|
| 64 | 2 | 109440 | 0.88 | 2.63 | 1.75 | 2.7 |
| 128 | 4 | 811008 | 6.49 | 19.46 | 12.98 | 21.2 |
| 192 | 6 | 2694528 | 21.56 | 64.67 | 43.11 | 47.7 |
| 256 | 8 | 6349824 | 50.80 | 152.40 | 101.60 | 84.7 |

## 2. Where the time goes

Cost breakdown of the default model (d = 128, L = 4, H = 4, d_ff = 512, T = 128, 811 k parameters) from `timed_forward` / `timed_backward` — copies of the library functions with `tic`/`toc` around each kind of work, means of 5 windows. "Sum of parts" against the uninstrumented time shows the instrumentation itself costs < 3 %.

| forward part | ms | % of fwd |
|--------------|----|----------|
| unpack + cache alloc | 3.80 | 7.6 |
| embedding gather | 0.09 | 0.2 |
| cache writes | 2.58 | 5.1 |
| LayerNorm fwd (gpt_ln_fwd) | 3.64 | 7.3 |
| QKV matmul | 6.75 | 13.5 |
| head slicing | 0.95 | 1.9 |
| scores Q K' + mask | 3.17 | 6.3 |
| softmax | 0.86 | 1.7 |
| A*V + column write | 2.42 | 4.8 |
| Wo matmul + residual | 2.43 | 4.9 |
| W1 matmul + bias | 9.98 | 19.9 |
| gelu | 2.01 | 4.0 |
| W2 matmul + bias + residual | 9.80 | 19.5 |
| LM head matmul | 0.57 | 1.1 |
| loss (softmax + scalar loop) | 0.58 | 1.1 |
| cache struct build | 0.55 | 1.1 |
| sum of parts | 50.16 | (uninstrumented forward 49.95 ms) |

| backward part | ms | % of bwd |
|---------------|----|----------|
| dlogits loop | 1.48 | 1.5 |
| unpack + grad alloc | 7.37 | 7.3 |
| LM head matmuls | 1.09 | 1.1 |
| LayerNorm bwd (gpt_ln_bwd) | 4.68 | 4.6 |
| cache slice reads | 4.70 | 4.6 |
| LN re-affine (gpt_rows) | 0.82 | 0.8 |
| FFN matmuls (W1, W2) | 35.21 | 34.7 |
| gradient writes | 2.92 | 2.9 |
| gelu grad | 12.25 | 12.1 |
| Wo / Wqkv matmuls | 17.69 | 17.4 |
| attention matmuls (dA dV dQ dK) | 9.46 | 9.3 |
| softmax bwd (elementwise) | 2.24 | 2.2 |
| embedding scatter-add | 1.22 | 1.2 |
| gradient struct build | 0.36 | 0.4 |
| sum of parts | 101.49 | (uninstrumented backward 102.07 ms) |

Matmul-bearing parts: 98.6 ms of 151.7 ms (65 %), running at 7.29 GFLOP/s inside those parts; everything else 53.1 ms. Over the whole window the 6·N·T convention gives 4.10 GFLOP/s.

**Primitives at the tutorial's shapes** (mean of 20 calls):

| primitive | shape | ms | GFLOP/s |
|-----------|-------|----|---------|
| layernorm builtin | 128 x 128 | 0.059 | |
| gpt_ln_fwd (library) | 128 x 128 | 0.322 | |
| softmax builtin | 128 x 128 | 0.058 | |
| gelu builtin | 128 x 512 | 0.335 | |
| matmul (128x128)(128x512) | | 2.043 | 8.21 |
| matmul (512x512)(512x512) | | 33.340 | 8.05 |
| attention core, one head | T=128 d_k=32 | 0.381 | 5.50 |
| embedding gather E(ids,:) | 129 rows of 128 | 0.022 | |
| embedding scatter-add loop | 129 rows of 128 | 1.185 | |
| scalar loop iteration | | 0.00017 | (0.17 us) |

**Per-step work outside the windows** (what `train.rlab` pays once per step on top of B windows):

| step component | ms |
|----------------|----|
| gpt_unflatten (flat -> struct) | 24.12 |
| batch reduction sum(Gm(:,1:n),1)/B | 34.44 |
| gpt_clip | 5.23 |
| gpt_adamw (flat vectors) | 36.46 |
| total per step | 100.25 |

At B = 8 that is 100 ms against a 0.37 s `parmap` call — about a quarter of every training step is spent outside the model, on 811 k-element flat vectors.

Reading the breakdown:

- **The dominant term is dense matmul, but it is not the whole story.** Matmul-bearing parts are 98.6 of 151.7 ms (65 %) of a forward + backward window and run at 7.3 GFLOP/s inside those parts (a bare matmul at the same shapes reaches 8.1–8.2 GFLOP/s); over the whole window the 6·N·T convention gives 4.1 GFLOP/s. Everything else — LayerNorm, GELU and its gradient, cache writes and slice reads, struct unpacking and allocation, the Python-style scalar loops — is the other third, and it is interpreter-bound, not FLOP-bound.
- **Real matmul pays for complex arithmetic.** `Value::Matrix` is `ndarray::Array2<Complex<f64>>` (`crates/rustlab-core/src/types.rs`), and `*` is `a.dot(b)` (`crates/rustlab-script/src/eval/value.rs:1920`), which ndarray routes to `matrixmultiply::zgemm` (single-threaded; rustlab enables no BLAS feature). A complex multiply-add is 4 real multiplies + 4 adds against 2 for a real one, over twice the bytes, so the measured 7–9 "real" GFLOP/s is really ≈ 30–35 GFLOP/s of complex arithmetic — a respectable fraction of one P-core's NEON f64 peak (≈ 16 flop/cycle × 4.4 GHz ≈ 70 GFLOP/s). The 4× arithmetic and 2× bandwidth penalty is paid on every product in the model for no reason: nothing in the tutorial is complex.
- **Comparison (estimate — numpy is not installed here to measure).** On this class of machine a single-threaded real dgemm from the same `matrixmultiply` crate typically reaches 40–55 GFLOP/s (≈ 6× today); Apple Accelerate's dgemm uses the AMX units and typically reaches 200–400 GFLOP/s f64 single-threaded and ≈ 1 TFLOP/s f64 across cores; sgemm (f32) roughly doubles those; OpenBLAS without AMX gives ≈ 40–60 GFLOP/s per core f64. Against today's 7–9 GFLOP/s that is 5× (same crate, real kernel) to 30–50× (Accelerate) on the matmul-bearing two thirds of the window.
- **The interpreter third.** Per-iteration overhead is ≈ 0.17 µs (0.13 µs in an earlier measurement today), so the scalar loops (loss, dlogits, scatter-add: ≈ 4 ms per window) are not the problem; the costly items are whole-matrix temporaries: `gpt_gelu_grad` builds six 128 × 512 intermediates (≈ 12 ms, 12 % of backward); the library LayerNorm `gpt_ln_fwd` is 5–8× slower than the `layernorm` builtin (0.32 vs 0.04–0.06 ms) because the affine broadcast goes through `outer` products (`gpt_rows`/`gpt_cols`); every `X_all(r, :) = X` cache write and `hid_all(r, :)` slice read copies (10 % of a window together); unpacking struct fields and allocating the caches is another 9 % of forward and 8 % of backward; `gpt_flatten` copies 811 k values through `reshape` + concatenation (10–14 ms, ≈ 7–9 % on top of a window). All of these are roadmap C11 territory (clone-free `Value` on hot paths).
- **What `--profile` sees.** Top of the stderr table for the benchmark run. Two limits make it blunt for this workload: operators are not calls, so `*`, indexing and indexed assignment never appear; and only the outermost user function called from the script is attributed — `time_one` (which wraps `gpt_forward` + `gpt_backward`) and `parmap` carry everything, while `gpt_forward`, `gpt_backward`, `gpt_ln_bwd` and the `softmax`/`gelu` calls inside them are absent from the table (the `gelu`/`softmax`/`gpt_ln_fwd` rows below are the top-level primitive loops only):

```
  Function         Calls    Total (ms)    Avg (µs)     In (KB)    Out (KB)      Mbit/s
  ────────────────────────────────────────────────────────────────────────────────────────
  parmap              20      9600.044  480002.177        1.98  2449441.98      2090.2
  time_one            21      8860.402  421923.915       38.33        0.00         0.0
  gpt_adamw           20       686.968   34348.381  1267200.94        0.00     15111.2
  timed_backward       5       563.274  112654.734       20.08        0.00         0.3
  gpt_unflatten       20       469.089   23454.433   253440.00        0.00      4426.0
  sum                 58       334.580    5768.626  2027529.09   253440.30     55848.2
  timed_forward        5       262.935   52586.900       20.08        0.00         0.6
  gpt_init             5       197.430   39486.017        0.04        0.00         0.0
  gpt_clip            20        92.724    4636.204   253440.16        0.00     22391.0
  gpt_flatten          1        11.048   11047.708        0.00    12672.00      9396.4
  savefig              1         8.306    8306.083        0.00        0.00         0.0
  gpt_ln_fwd          20         6.377     318.862     5200.00     5120.00     13256.7
  gelu                20         6.334     316.692    20480.00    20480.00     52976.5
  randn                9         3.697     410.806        0.14     7004.00     15519.1
```

## 3. What a Sherlock-scale run costs today

- **Corpus:** 562 269 characters, 77-symbol vocabulary; 534 156 training tokens per epoch (5 % held out). Default run: 3 000 steps × 8 × 128 = 3.07 M tokens = 5.75 epochs.
- **Minutes per epoch at measured rates (default 811 k model):** serial ≈ 11 min; `parmap` B = 8 ≈ 2.2 min; the real run's wall-clock rate including the per-step optimiser work, evaluation and checkpoints gives ≈ 3.5 min per epoch, so the whole 5.75-epoch run is ≈ 20 min. The run reached 2.59 bits/char held-out at 2.9 epochs and 2.20 bits/char at the end.
- **What fits in a time budget** (parmap(8) rates from the sweep; a 0.5 M-character corpus wants roughly 5–15 epochs before it overfits, so "bigger" past ~3 M parameters buys nothing without more text):

| d | L | params | parmap(8) tok/s | epochs in 10 min | epochs in 1 h | epochs in 8 h |
|---|---|--------|-----------------|------------------|---------------|---------------|
| 64 | 2 | 109440 | 16398 | 18.4 | 110.5 | 884 |
| 128 | 4 | 811008 | 4122 | 4.6 | 27.8 | 222 |
| 192 | 6 | 2694528 | 1351 | 1.5 | 9.1 | 73 |
| 256 | 8 | 6349824 | 456 | 0.5 | 3.1 | 25 |

- **What changes with each request below.** (1) BLAS-backed real matmul removes most of the matmul two thirds: the window drops from ≈ 152 ms to ≈ 56–73 ms (≈ 2.1–2.7×) and the interpreter third becomes the ceiling. (4), (6) and the builtins in (7) attack that third; together with (1) a window of ≈ 20–25 ms is realistic (≈ 7×, ≈ 6 000 tok/s serial, 30–40 k tok/s with `parmap`), i.e. an epoch of the default model in ≈ 15 s, and the 6.3 M model's overnight budget (25 epochs) shrinks to about an hour. (2) matters only after (1): threads inside dgemm and `parmap` workers compete for the same 14 CPUs, so it is one or the other per call site. (3) removes the Python step, not time. (5) removes ≈ 150 lines of hand-written backward code and the activation cache, not time.

## 4. Requests to rustlab, ranked by measured impact

Format follows `AGENTS.md` → "Rustlab Recommendations". Roadmap IDs refer to `../rustlab/dev/plans/roadmap.md`.

### `*` on real matrices -> BLAS dgemm (real-valued storage or a realness flag); optional `single(X)` -> sgemm
**Needed for:** every matmul in `lib/gpt.rlab` (and Lessons 13–18, 22–23) — the 65 % of window time measured above.
**Purpose:** Dispatch `A * B` to a real GEMM when both operands have zero imaginary parts: Accelerate on macOS (`ndarray` `blas` feature + `blas-src = accelerate`), OpenBLAS elsewhere, `matrixmultiply::dgemm` as the no-dependency fallback. `single(X)` / `double(X)` would select f32 storage for a further ≈ 2× where the user accepts it (this tutorial's gradients are fine in f32; AdamW state wants f64).
**Current state (0.3.9):** storage is `Array2<Complex<f64>>`; `*` is `matrixmultiply::zgemm`, single-threaded, 7–9 real-GFLOP/s at every shape tried (128×128·128×512, 512×512, 1024×256·256×256). rustlab's own `dev/plans/em_requests_plan.md` names this as "Option C — fully real-typed storage … real perf wins (dgemm, real FFT)", long-term; this document is the measured case for it. Interim: a `Value::Matrix { data, is_real }` flag (its Option B) is enough to pick the real kernel, because the kernel choice — not the storage — is the 4× arithmetic penalty.
**Example:** `logits = Hf * W_U;` — unchanged source; `theta32 = single(theta);` opt-in.
**Expected gain:** matmul-bearing parts ÷ 5 (dgemm, same crate) to ÷ 30 (Accelerate); whole window ≈ 2.5× today, up to ≈ 7× once the interpreter third is also addressed.

### `pagemtimes(A3, B3)` over Tensor3, or threads inside `*`
**Needed for:** the per-head loop (H independent T×T products per layer, forward and 4 per head in backward) and the batch of B windows in `train.rlab`.
**Purpose:** One call multiplying matching pages of two rank-3 arrays — `O = pagemtimes(A, V3)` with `A` T×T×H and `V3` T×d_k×H — run across cores, so the head loop and (with a batched forward) the window loop stop being interpreter loops over small matrices. Alternatively a multithreaded GEMM inside `*` for large operands (Accelerate does this by itself above a size threshold).
**Current state (0.3.9):** `parmap` is the only parallelism; it is a rayon pool that re-enters the interpreter per task, clones the captured parameter struct (13 MB complex) into each worker and returns a 13 MB gradient row per window. Measured: speedup 1.7× at B = 2, 2.6× at B = 4, 3.4× at B = 8 and 3.9× at B = 14 in the scaling table (28 % efficiency at the top); the sweep's parmap(8) column saw 5.1× at B = 8. Tensor3 exists (`CTensor3`) but has no matmul.
**Example:** `O = pagemtimes(A, Vh3);  S = pagemtimes(Q3, permute(K3, [2, 1, 3])) * scale;`
**Expected gain:** heads: 4 (default) to 8 (large config) small matmuls plus slicing per layer collapse into one call; batch: the parmap efficiency lost between B = 4 and B = 14 (efficiency falls from 83 % at B = 2 to 28 % at B = 14) is recovered when the batch is one tensor instead of B interpreter tasks. Note the interaction with (1): dgemm threads and `parmap` workers share the 14 CPUs.

### Text I/O and string<->code conversion — `fileread(path)`, `double(str)`, `char(codes)`, escapes in literals, `sa(i) = "x"`
**Needed for:** `tutorial/sherlock/prepare_corpus.py` exists only because of this; `gpt_encode`/`gpt_decode` in `lib/gpt.rlab` are O(len × V) character-compare loops; `data/vocab.rlab` has to spell newline and quote as `sprintf("%c", 10)` / `sprintf("%c", 34)`.
**Purpose:** Read a text file into a string; convert a string to its code points and back; let `"\n"`, `"\t"`, `"\""`, `"\\"` mean what they mean in `sprintf` already; allow assignment into a string array element.
**Current state (0.3.9), all reproduced today:**
- `print("a\nb")` prints the four characters `a\nb` — no escapes in plain literals (`sprintf("a\nb")` does interpret them).
- `double("A")` → `type error: double: cannot convert string`; `char(65)` → `undefined function 'char'`; `fileread(...)` → `undefined function 'fileread'`; `upper`, `unique` also undefined.
- `sa = {"a", "b", "c"}; sa(2) = "z"` → `runtime error: index assignment: right-hand side must be scalar or complex, got string`.
- Works today: `s(i)` indexes a character, `s + t` concatenates, `length(s)`, `sprintf("%c", n)`.
**Example:**
```
txt   = fileread("data/adventures.txt");      # one string
codes = double(txt);                          # 1 x N code points
vocab = unique(codes);                        # sorted distinct codes
ids   = lookup(vocab, codes);                 # 1-based ids (or arrayfun over find)
text  = char(vocab(ids(1:300)));              # back to a string
```
**Expected gain:** the tutorial becomes pure rustlab (one less toolchain for the student); `gpt_decode` of a 300-character sample stops concatenating strings in a loop; the vocabulary file becomes a plain `save`/`load` of codes.

### Fused causal attention — `[O, A] = attention(Q, K, V, "causal")` and `[dQ, dK, dV] = attention_bwd(dO, Q, K, V, A)`
**Needed for:** `lib/gpt.rlab` head loops: forward slots "head slicing / scores / softmax / A·V" = 7.4 ms, 15 % of forward; backward "attention matmuls + softmax bwd" = 11.7 ms, 12 % of backward; plus the `A_all` cache (L·H·T × T, 4.2 MB per window at T = 128, 16 B per entry).
**Purpose:** One builtin does `softmax(Q Kᵀ / √d_k + causal mask) V` per head (heads along the third axis or as column blocks) and its backward, with the mask applied inside the kernel rather than as an added `-1e9` matrix, and no T×T temporaries surfacing in the interpreter.
**Current state (0.3.9):** composed from `*`, `softmax`, an explicit additive mask and column slicing; one head at T = 128, d_k = 32 costs 0.42 ms forward (5 GFLOP/s) and the surrounding slicing/writes as much again; at T = 256 it is 1.35 ms per head.
**Example:** `[O, A] = attention(QKV(:, 1:d), QKV(:, d+1:2*d), QKV(:, 2*d+1:3*d), "causal", H);`
**Expected gain:** ≈ 3–4× on the attention share (≈ 10 % of a window today, a larger share once (1) lands because attention does not scale with d_ff), and the option of not caching A at all.

### Reverse-mode autodiff — `[L, G] = grad(f, P)`
**Needed for:** already the top open request in `AGENTS.md` (Lessons 15, 22, 24); not listed in rustlab's `roadmap.md`. In `lib/gpt.rlab` it would delete `gpt_backward` (≈ 80 lines), `gpt_dlogits`, `gpt_ln_bwd`, `gpt_gelu_grad`, the 11 stacked cache matrices (21 MB per window at the default size, 85 MB at d = 256) and the finite-difference self-test in `test_gpt_lib.rlab` — about 150 of 460 lines, and every line that has to be re-derived when the architecture changes (SFT, DPO, a RoPE or GQA variant).
**Purpose:** A tape over matrix ops (`*`, `.*`, `+`, `softmax`, `layernorm`, `gelu`, indexing, `sum`, `log`) with the reverse pass for each.
**Current state (0.3.9):** only grid gradients (`gradient`, `gradient3`).
**Example:** `[L, G] = grad(@(P) gpt_loss(ids, P, cfg), P);  g = gpt_flatten(G, cfg);`
**Expected gain:** correctness and maintenance rather than speed (a tape costs about what the hand-written pass costs); it also makes (6) automatic, because the tape owns the accumulation.

### In-place and accumulating indexed ops — `M(r, :) += X`, `s.f += X`, no-copy `M(r, :) = X`
**Needed for:** cache writes (5 % of forward), cache slice reads (5 % of backward), per-layer gradient writes (3 %), the embedding scatter-add `dE(ids(t), :) = dE(ids(t), :) + dX(t, :)` (1.2 ms per window, a read-modify-write copy per row), `gpt_flatten` (10–14 ms) and the per-step `gpt_unflatten` + batch reduction + AdamW on the flat vector (100 ms per step).
**Purpose:** Let an indexed assignment mutate the stored array instead of rebuilding it, and accept the compound forms on indexed and struct-field targets.
**Current state (0.3.9):** `M(2:3, :) += X` and `s.a += 1` are both `parse error: expected newline or EOF, got PlusEq`; plain `M(r, :) = X` works but copies. Upstream plan C11 (`dev/plans/eval_perf_value_borrow.md`, "stop cloning `Value` on hot paths") covers the reads; the compound-assignment syntax is new.
**Example:** `dE(ids(t), :) += dX(t, :);   Hin_all(r, :) = X;   G.W2 += hid' * dX;`
**Expected gain:** ≈ 15–20 % of a window today (writes + reads + unpack/alloc), most of the flatten, and the bulk of the per-step optimiser overhead — which is why the real run averages 2,527 tok/s against 4,122 for the `parmap` call alone.

### Smaller items the profile and the breakdown point at
- **`gelu_grad(z)` (or `[y, dy] = gelu(z)`):** `gpt_gelu_grad` is 13 ms = 12 % of backward against 0.35 ms for the `gelu` builtin on the same 128 × 512 matrix — a builtin derivative is a direct ≈ 10× on that line.
- **`layernorm(X, gamma, beta)` and `layernorm_bwd(dY, X, gamma)`:** the affine form is what every transformer needs; the library's `gpt_ln_fwd` is 5–8× the builtin because `gamma`/`beta` have to be broadcast with `outer`. Together LayerNorm is 7 % of forward and 5 % of backward.
- **Vector orientation:** `reshape(v, n, 1)` of a vector collapses back to a bare vector, so `repmat(reshape(v, n, 1), 1, d)` tiles *sideways* (`[1×12]` for n = 4, d = 3); the library had to use `outer` for every broadcast. Related to roadmap C12 (`zeros(n)` is 1×n). A column vector that stays a column, or a `broadcast(v, T, 1)` builtin, would remove `gpt_rows`/`gpt_cols`.
- **Profiler attribution:** `--profile` cannot see operators, indexing or indexed assignment, and it does not descend into nested user functions; a "self" row per function at every depth, with `*` counted as a call, would have made this study a one-liner.
- **`parmap` payload:** each task captures `P` by value (13 MB complex at the default size, 100 MB at d = 256); a shared read-only capture (Arc) would remove the per-task copy.

## 5. What the tutorial does regardless, and the headroom

`lib/gpt.rlab` keeps the design that is right whatever rustlab does next: a single flat f64 parameter vector (so AdamW, clipping, checkpoints and the parmap gradient sum are one-liners), stacked per-layer matrices so there is one matmul per sublayer rather than a loop over heads where it can be avoided, `parmap` data parallelism over the batch (the measured 3.4–5.1× at B = 8 is the only multi-core path today), and f64 throughout (f32 is a gift the student accepts later, not a correctness requirement). Measured headroom on this machine, in order: ≈ 2.5× from real BLAS matmul alone, ≈ 7× with the interpreter third also addressed, and a further 3–5× from using all cores inside one window instead of across windows — roughly two orders of magnitude between today's ≈ 810 serial / 2,527 end-to-end tok/s and what the same hardware does in numpy-with-Accelerate.
