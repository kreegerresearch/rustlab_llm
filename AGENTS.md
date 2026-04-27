# AGENTS.md

This file guides AI coding tools working in this repository.

## Where to Start

1. Read `PLAN.md` for current phase status and handoff notes.
2. Read this file for project conventions, the Rustlab language reference, and content format rules.
3. Check the target phase's **Handoff Notes** in `PLAN.md` before starting work.
4. Update `PLAN.md` when completing a lesson or pausing mid-phase.

---

## Project Purpose

`rustlab_llm` is a self-contained tutorial series for building large language models from first principles, using [Rustlab](../rustlab) as the scripting and visualisation environment.

Each lesson pairs step-by-step mathematical theory with runnable Rustlab scripts and integrated notebooks that produce visualisations. The series builds from raw probability and linear algebra to a complete GPT-style decoder — nothing is a black box.

**Learning goal:** Derive every core LLM algorithm — tokenisation, attention, transformer blocks, training, and inference — with working code and plots at each step, following the architecture of [nanoGPT](https://github.com/karpathy/nanoGPT).

**Prerequisites:** Linear algebra, basic probability and information theory. No deep learning background required.

This repo is **independent from rustlab** — never modify `../rustlab` from here. If a needed function is missing, add a workaround in the script with a `# TODO: replace with built-in <name> once available` comment and note it in the Rustlab Recommendations section below.

---

## Repository Layout

This repo follows the [Rustlab lesson-site pattern](../rustlab/docs/lesson-site-pattern.md) — sources flat in `notebooks/`, rendered output committed to a top-level `site/`, optional `.r` scripts in `lessons/<slug>/`.

```
notebooks/
  README.md              # editor-facing notes (skipped by renderer)
  NN-topic-slug.md       # source notebooks — prose + math + ```rustlab``` blocks

lessons/
  README.md              # explains the .r-script convention
  NN-topic-slug/
    *.r                  # standalone shell-runnable rustlab scripts
    *.svg|*.png|*.html   # script artefacts (gitignored)

site/                    # rendered output for GitHub display
  README.md              # hand-written GitHub landing page
  NN-topic-slug.md       # rendered notebook with inline ![](plots/...) SVGs (committed)
  plots/NN-topic-slug/   # captured figures (committed)
  index.html             # auto-generated entry page (gitignored)
  NN-topic-slug.html     # interactive Plotly per-notebook (gitignored)

PLAN.md                  # phase status, handoff notes
README.md                # project overview + lesson roadmap
Makefile                 # notebooks / html / lesson-NN / clean
```

There is no per-lesson `lesson.md` — the notebook *is* the lesson (theory, code, plots, exercises in one file). When `.r` scripts mirror notebook code blocks, they live under `lessons/<slug>/` and call `savefig("foo.svg")` next to themselves (artefacts gitignored).

---

## Running

All commands run from the project root:

```bash
make                    # show help
make all                # render committed site/<slug>.md + interactive site/*.html
make notebooks          # render site/<slug>.md from notebooks/<slug>.md (markdown)
make html               # render site/index.html + per-notebook html (gitignored)
make notebooks-check    # CI drift guard: fails if site/ is out of sync with sources
make lesson-01          # run only lesson 01's .r scripts (works for 01–09)
make clean              # delete the interactive HTML build and .r artefacts
```

The notebook render is directory-mode: `rustlab notebook render notebooks --format markdown --output site` produces `site/<slug>.md` plus `site/plots/<slug>/plot-N.svg` for each lesson. The hand-written `site/README.md` is preserved (the renderer skips files named `README.md` on input).

Single script:
```bash
rustlab run lessons/01-tokens-and-encoding/char_frequencies.r
```

Interactive REPL:
```bash
rustlab
```

---

## Notebook Format (`notebooks/<slug>.md`)

Use GitHub-flavored Markdown with LaTeX math: `$inline$` and `$$block$$`. Each notebook follows this structure:

1. `# Lesson NN: Title` — H1, no suffix
2. Brief motivation paragraph
3. `## Learning Objectives` — 3–5 bullets
4. `## Background` — prerequisite knowledge assumed for this specific lesson
5. Pure-reference H2s when needed (formal definitions, vocabulary tables, dimension conventions) — these stay flat with no H3 split
6. One H2 per concept, each split into `### Theory` (prose + math, no code) and one or more `### Example — <descriptor>` (rustlab block plus a short setup paragraph). One H3 per logically distinct example — if a concept has both a frequency bar chart and a one-hot heatmap, each gets its own `### Example — ...`. The H3 markers should always be present so readers can tell theory from examples at a glance; only genuinely all-reference sections keep flat H2s
7. `## Key Takeaways` (optional) — short summary
8. `## Standalone Scripts` — table referencing the parallel `.r` files
9. `## Expected Numerical Outputs Summary` — Markdown table of every `print()` value students should see
10. `## Exercises` — 3–5 follow-up questions or script modifications
11. `## What's next` — one paragraph forward link to the next lesson

**Authoring rules (renderer-specific):**

- Variables persist across ` ```rustlab ` blocks within a notebook — define `[X, Y]`, `vocab_size`, etc. once, reuse below.
- The renderer captures the active figure automatically. **Don't** call `savefig()` inside notebook code blocks.
- Use `figure()` (not `clf;`) at the start of each plot block. `figure()` creates a fresh figure and avoids state leaking across notebooks in directory-mode rendering.
- If a plot block uses `hold("on")`, close it with `hold("off")` at the end. Lingering `hold("on")` state leaks into the next notebook in directory mode and inflates the captured-plot count.
- Use `<!-- hide -->` before setup-only code blocks the reader doesn't need to see.
- Use template interpolation `${expr}` to embed computed values in prose (e.g. `${mean(v):%.3f}`).
- Comments in notebook code blocks: `%`. Comments in `.r` files: `#`.

**Style:**

- Derive equations step-by-step; never skip a step without explanation.
- Name every variable and state units explicitly.
- Connect math to intuition: explain *why* the result looks the way it does.
- Call out common misconceptions explicitly.

---

## Script Conventions (`.r` files)

**Required header block:**
```r
# Script:  [filename].r
# System:  [what system is being modeled]
# Concept: [the single concept this script demonstrates]
# Equations: [key equations, in plain text]
# Units:   [what units are used for each quantity]
```

**Rules:**
- Separate logical sections with `# === Section Name ===`
- Plot output saves next to the script: `savefig("foo.svg")` (no `outputs/` prefix). The artefact is gitignored.
- Always `print()` key numerical results a student should verify by hand
- Name files descriptively: `gradient_descent.r`, not `script1.r`
- Keep scripts short enough to read in one sitting (split if over ~60 lines)
- Each script must run independently (no shared state between scripts)

---

## Rustlab Language Reference

Rustlab is a scientific computing CLI (`../rustlab`) with a MATLAB-like scripting language. Full reference: `../rustlab/docs/quickref.md`. Function signatures and examples: `../rustlab/docs/functions.md`. Notebook spec: `../rustlab/docs/notebooks.md`.

### Language Essentials

- 1-based indexing: `v(1)` is the first element; `v(end)` is the last
- Suppress output with `;`; comment with `#` or `%`
- Element-wise ops: `.^`, `.*`, `./`; matrix multiply: `*`; conjugate transpose: `'`
- Column vector: `v = [a; b; c]`; matrix: `M = [a, b; c, d]`
- Range: `1:10`, `0:0.1:1`, `10:-1:1`
- For loop: `for i = 1:n` ... `end`
- While loop: `while cond` ... `end`
- Conditionals: `if` / `elseif` / `else` / `end`
- Switch: `switch expr` / `case val` / `otherwise` / `end`
- Functions: `function [out] = name(args)` ... `end`
- Anonymous functions: `@(x) x.^2`; function handles: `@name`
- Chain indexing: `f(args)(i)` works without a temporary
- Compound assignment: `+=`, `-=`, `*=`, `/=`
- Line continuation: `...`
- Destructuring: `[X, Y] = meshgrid(x, y)`
- Indexed assignment grows vectors: `v(i) = val`
- String arrays: `{"a", "b", "c"}`
- Structs: `s.field = val` auto-creates struct
- `clear` removes all variables; `clf` clears current figure

### Function Reference (subset relevant to this tutorial)

**Math (element-wise):** `exp`, `log`, `log2`, `log10`, `sqrt`, `abs`, `sin`, `cos`, `asin`, `acos`, `atan`, `atan2`, `tanh`, `sinh`, `cosh`, `real`, `imag`, `conj`, `angle`, `floor`, `ceil`, `round`, `mod`, `sign`

**Statistics:** `sum`, `prod`, `cumsum`, `min`, `max`, `argmin`, `argmax`, `mean`, `median`, `std`, `sort`, `trapz`, `hist(v,n)`, `all`, `any`

**ML / Activations:** `softmax(v)`, `relu(v)`, `gelu(v)`, `layernorm(v)`, `layernorm(v, eps)`

**Array construction:** `zeros(n)` / `zeros(m,n)`, `ones(n)` / `ones(m,n)`, `eye(n)`, `linspace(a,b,n)`, `logspace(a,b,n)`, `rand(n)`, `randn(n)` / `randn(m,n)`, `randi(imax,n)`, `randi([lo,hi],n)`

**Array inspection:** `len(v)`, `length(v)`, `numel(x)`, `size(x)`

**Matrix ops:** `reshape(A,m,n)`, `repmat(A,m,n)`, `transpose(A)`, `diag(v)` / `diag(M)`, `outer(a,b)`, `kron(A,B)`, `inv(M)`, `linsolve(A,b)`, `det(M)`, `trace(M)`, `rank(M)`, `eig(M)`, `svd(A)`, `expm(M)`, `norm(v)` / `norm(v,p)`, `dot(u,v)`, `cross(u,v)`, `meshgrid(x,y)`, `roots(p)`

**Concatenation:** `horzcat(A,B,...)` / `[A, B]`, `vertcat(A,B,...)` / `[A; B]`

**Structs:** `struct("k",v,...)`, `s.field`, `s.field = val`, `isstruct(x)`, `fieldnames(s)`, `isfield(s,"name")`, `rmfield(s,"name")`

**String arrays:** `{"a","b","c"}`, `sa(i)`, `iscell(x)`, `length(sa)`, `numel(sa)`

**Higher-order:** `arrayfun(f, v)`, `feval("name", args...)`

**I/O:** `print(x,...)`, `disp(x)`, `fprintf(fmt,...)`, `sprintf(fmt,...)`, `commas(x)`, `save(file,x)`, `save(file,"name",x,...)` (NPZ), `load(file)`, `load(file,"name")`, `whos`

**Plotting (primary API — interactive plot + file save):**
```
plot(v)  /  plot(x, y, "color", "blue", "label", "name", "style", "dashed")
bar(y)  /  bar(labels, y)  /  bar(M)        — bar / categorical / grouped
scatter(x, y)
imagesc(M, "viridis")                        — heatmap (colormaps: viridis, jet, hot, gray)
[X, Y] = meshgrid(x, y)                      — coordinate matrices (size length(y) × length(x))
surf(Z)  /  surf(X, Y, Z)  /  surf(X, Y, Z, "viridis")  — 3D surface (rotatable HTML, static SVG/PNG)
histogram(v)
savefig("file.svg")                          — save current figure to SVG, PNG, or HTML
```

**Figure controls:**
```
figure()  /  figure("file.html")
subplot(rows, cols, idx)
hold("on")  /  hold("off")
grid("on")  /  grid("off")
title("text")  /  xlabel("text")  /  ylabel("text")
xlim([lo, hi])  /  ylim([lo, hi])
hline(y, "color", "label")                   — horizontal reference line
legend("s1", "s2")
clf                                          — clear current figure
```

**Canonical save pattern.** The shorthand `savebar`, `savescatter`, `saveimagesc`, and `savehist` wrappers are deprecated — use the `plot/bar/scatter/imagesc` call followed by `savefig(file)`:

```
figure()
bar(y, "title")                    % or: scatter(x, y, "title")
savefig("outputs/chart.svg")

figure()
imagesc(M, "viridis")
title("Heatmap")
savefig("outputs/heatmap.svg")
```

---

## Rustlab Recommendations

When a needed function is missing from rustlab, record it here with the format:

```
### function_name(args) -> return_type
**Needed for:** Lesson NN — [title]
**Purpose:** [what it computes]
**Example:** result = function_name(arg1, arg2)
```

### ✅ `seed(n)` — **landed in rustlab** (commit `2bf8156`)
**Was needed for:** Lessons 04 (embeddings) and 05 (bigram sampling), and any later lesson that uses `rand()` / `randn()`.
**Use:** `seed(N)` sets the global RNG to a deterministic state; subsequent `rand` / `randn` / `randi` / `sprand` calls are bit-stable. `seed()` (no argument) re-randomizes from system entropy.
**Migrated:** Lessons 04 and 05 originally used a sin/cos-based pseudo-random matrix and a hand-set `draws` vector with TODO markers; both have been updated to use `seed(N)` followed by `randn` / `rand`.
**Example:**
```
seed(42);
E = randn(8, 6) * 0.1;   % bit-identical across runs
```

### `M(idx)` row gather with an integer-vector index → matrix
**Needed for:** Lesson 11 (FFN per-position independence check) and any later lesson that wants to permute, gather, or sample a subset of rows of a matrix.
**Current behaviour:** `M(2)` returns row 2, `M(2, 3)` returns the scalar at (2, 3), but `M([3, 1, 2])` raises `runtime error: matrix single-index with vector not supported; use M(i,j) for element access`.
**Workaround in use:** Build a permutation/selection matrix `P` and compute `P * M`. Works correctly but is allocation-heavy ($N^2$ memory for an $N$-row gather) and requires the user to construct the permutation matrix by hand. See `lessons/11-feed-forward-block/ffn_forward.r` for the worked pattern.
**Wanted:** Standard MATLAB/Octave-style row gather, accepting an integer vector or range and returning the gathered rows.
**Example (target):**
```
H_perm = H([3, 1, 2, 5, 4]);          % gather rows in any order
batch  = X(rand_indices);             % minibatch sampling for SGD lessons (Phase 6)
```

### Vector vs. 1×N matrix type distinction in arithmetic
**Hit while writing:** Lesson 12 (residual signal demo).
**Symptom:** `(W * x')'` and `x * W'` are mathematically identical (both produce a row of length $d$), but the first returns a `matrix` of shape $1 \times d$ while the second returns a `vector`. Adding a `vector` to a `matrix` raises `type error: operator Add not defined for vector and matrix`. Surfaces whenever you mix the two patterns in a per-step update like `x = x + alpha * f(x)`.
**Workaround in use:** Always project via `x * W'` (or use `reshape(M, 1, d)` to coerce a 1×$d$ matrix back to a vector) so types stay aligned. See `lessons/12-layer-norm-and-residuals/residual_signal.r`.
**Wanted:** Either treat `1 × N` matrices as auto-promotable to vectors for `+`/`-`, or — more conservatively — accept vector + 1×N-matrix and broadcast. Without one of these, intermediate transposes silently change a value's type and the error appears far from the root cause.
**Example (target):**
```
y = (W * x')';             % current: returns matrix
z = x + 0.1 * gelu(y);     % current: errors; want: just works
```

### `layernorm(M)` row-wise on a matrix → matrix
**Needed for:** Lesson 12 (LayerNorm sublayer) and every later transformer lesson — every transformer block applies LN per token vector to the full $T \times d_{\text{model}}$ residual stream.
**Current behaviour:** `layernorm(v)` works on a vector or scalar; `layernorm(M)` raises `type error: layernorm: argument must be a non-empty vector or scalar`.
**Workaround in use:** Loop per row — `for t = 1:T; LN(t) = layernorm(X(t)); end` — leveraging that `M(t) = vec` assigns a row. Correct and readable, but $O(T)$ scalar dispatches instead of one vectorised call. See `lessons/12-layer-norm-and-residuals/layernorm_distribution.r`.
**Wanted:** Matrix overload that normalises each row independently, matching the convention every transformer uses.
**Example (target):**
```
H_normed = layernorm(H);                    % shape (T, d_model), per-row mean=0 std=1
H_affine = layernorm(H, eps, gamma, beta);  % optional learned scale + bias
```
