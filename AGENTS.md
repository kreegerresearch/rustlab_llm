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

This repo follows the [Rustlab lesson-site pattern](../rustlab/docs/lesson-site-pattern.md) — sources flat in `notebooks/`, rendered output committed to a top-level `book/`, optional `.rlab` scripts in `lessons/<slug>/`.

```
notebooks/
  README.md              # editor-facing notes (skipped by renderer)
  NN-topic-slug.md       # source notebooks — prose + math + ```rustlab``` blocks

lessons/
  README.md              # explains the .rlab-script convention
  NN-topic-slug/
    *.rlab                  # standalone shell-runnable rustlab scripts
    *.svg|*.png|*.html   # script artefacts (gitignored)

book/                    # rendered output for GitHub display
  README.md              # hand-written GitHub landing page
  NN-topic-slug.md       # rendered notebook with inline ![](plots/...) SVGs (committed)
  plots/NN-topic-slug/   # captured figures (committed)
  index.html             # auto-generated entry page (gitignored)
  NN-topic-slug.html     # interactive Plotly per-notebook (gitignored)

PLAN.md                  # phase status, handoff notes
README.md                # project overview + lesson roadmap
Makefile                 # notebooks / html / lesson-NN / clean
```

There is no per-lesson `lesson.md` — the notebook *is* the lesson (theory, code, plots, exercises in one file). When `.rlab` scripts mirror notebook code blocks, they live under `lessons/<slug>/` and call `savefig("foo.svg")` next to themselves (artefacts gitignored).

---

## Running

All commands run from the project root:

```bash
make                    # show help
make all                # render committed book/<slug>.md + interactive book/*.html
make notebooks          # render book/<slug>.md from notebooks/<slug>.md (markdown)
make html               # render book/index.html + per-notebook html (gitignored)
make notebooks-check    # CI drift guard: fails if book/ is out of sync with sources
make lesson-01          # run only lesson 01's .rlab scripts (works for 01–09)
make clean              # delete the interactive HTML build and .rlab artefacts
```

The notebook render is directory-mode: `rustlab notebook render notebooks --format markdown --output book` produces `book/<slug>.md` plus `book/plots/<slug>/plot-N.svg` for each lesson. The hand-written `book/README.md` is preserved (the renderer skips files named `README.md` on input).

Single script:
```bash
rustlab run lessons/01-tokens-and-encoding/char_frequencies.rlab
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
8. `## Standalone Scripts` — table referencing the parallel `.rlab` files
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
- Comments in notebook code blocks: `%`. Comments in `.rlab` files: `#`.

**Math escaping for GitHub + Obsidian compatibility** (full reference in `../rustlab/docs/notebooks.md`):

- `${expr}$` in plain text auto-wraps as `$<value>$` (math-wrap shorthand). Inside an open `$...$` span, write `$X = ${expr}$` — the value emits bare and the trailing `$` closes the span.
- **In tables, use `\lvert ... \rvert`** (not raw `|...|`) for cardinality / absolute value. Raw `|` inside `$...$` splits the table cell on GitHub. Same for `\lVert ... \rVert` for norms.
- `\$` is the literal-`$` escape (currency). It does not toggle the math tracker, so `\$5 plus ${tax}$` works.
- `$$display$$` math should sit on its own paragraph line.

**Obsidian-aligned markdown features** (render natively on GitHub *and* Obsidian — full reference in `../rustlab/docs/notebooks.md`):

- **Callouts** — prefer `> [!NOTE]` / `[!TIP]` / `[!IMPORTANT]` / `[!WARNING]` / `[!CAUTION]` blockquote syntax. Optional inline title: `> [!TIP] Heads up`. The legacy `<!-- note -->` form still parses; on the next `make notebooks` it auto-migrates to GFM-native syntax in the rendered output.
- **Footnotes** — `[^id]` inline reference, `[^id]: text` definition.
- **Task lists** — `- [ ]` and `- [x]` render as checkboxes.
- **Explicit heading IDs** — `## Section {#stable-anchor}` pins a cross-notebook anchor.
- **Wikilinks** — `[[02-probability-and-softmax]]`, `[[02-probability-and-softmax|softmax lesson]]`, `[[02-probability-and-softmax#Temperature Scaling]]`. The renderer transforms them to ordinary markdown links (target gets `.md` appended for notebook refs); GitHub and Obsidian both render the result natively.
- **Embeds** — `![[diagram.svg]]`, `![[chart.png|alt text]]` for inline images. Path passes through as-is.

**Style:**

- Derive equations step-by-step; never skip a step without explanation.
- Name every variable and state units explicitly.
- Connect math to intuition: explain *why* the result looks the way it does.
- Call out common misconceptions explicitly.

---

## Script Conventions (`.rlab` files)

**Required header block:**
```r
# Script:  [filename].rlab
# System:  [what system is being modeled]
# Concept: [the single concept this script demonstrates]
# Equations: [key equations, in plain text]
# Units:   [what units are used for each quantity]
```

**Rules:**
- Separate logical sections with `# === Section Name ===`
- Plot output saves next to the script: `savefig("foo.svg")` (no `outputs/` prefix). The artefact is gitignored.
- Always `print()` key numerical results a student should verify by hand
- Name files descriptively: `gradient_descent.rlab`, not `script1.rlab`
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
heatmap(M)  /  heatmap(M, "title")           — heatmap with numeric axes
heatmap(xlabels, ylabels, M [, "title" [, "viridis"]])  — heatmap with categorical axis labels (row 0 at top)
[X, Y] = meshgrid(x, y)                      — coordinate matrices (size length(y) × length(x))
surf(Z)  /  surf(X, Y, Z)  /  surf(X, Y, Z, "viridis")  — 3D surface (rotatable HTML, static SVG/PNG)
histogram(v)
savefig("file.svg")                          — save current figure to SVG, PNG, or HTML
```

Use `heatmap(xlabels, ylabels, M, ...)` instead of `imagesc(M, ...)` whenever the matrix has categorical row/column meanings (vocabulary tokens, token positions, head/dim names) — the labels turn the heatmap into a direct lookup. Reach for `imagesc` for purely numeric matrices (loss landscapes, positional-encoding `pos × dim`, hidden activations).

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

### CLI should announce itself as the `.rlab` handler
**Needed for:** language identity. Lessons in this repo use the `.rlab` extension, but `rustlab run foo.rlab` produces output indistinguishable from running a `.r` or unsuffixed file — there is no banner or log line that identifies rustlab as the handler.
**Why it matters:** rustlab is a distinct DSP-modelling language; while we currently borrow MATLAB syntax highlighting as a temporary proxy (see `.gitattributes` and the README "Environment & Tooling" section), the CLI is the most authoritative place to reinforce that `.rlab` is *rustlab*, not MATLAB. A one-line banner at startup ("rustlab 0.1.x — running foo.rlab") gives every shell-pasteable command a clear provenance.
**Wanted:** an opt-in `--banner` flag, or an unconditional one-line stderr log on `rustlab run`, identifying the version + the file being executed. Optionally a stricter mode that warns when the input doesn't carry the `.rlab` extension.
**Example (target):**
```
$ rustlab run lessons/01-tokens-and-encoding/char_frequencies.rlab
rustlab 0.1.12 — running char_frequencies.rlab
Corpus: to be or not to be
...
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

### ✅ Vector + 1×N matrix arithmetic — **landed in rustlab 0.2.0** (commit `31ff3e3`)
**Was hit while writing:** Lesson 12 (residual signal) and every subsequent lesson that mixes a `vector` and a `1×N matrix` in arithmetic.
**Now works:** `vec + 1×N_matrix`, `vec .* 1×N_matrix`, etc. — implicit broadcasting promotes both sides. The `(W * x')'` returns-a-matrix path no longer breaks per-step updates like `x = x + alpha * f(x)`.
**Status of existing workarounds:** the lessons still use the explicit `M(1)` row-extract (`dL_da1_m(1)`) and `x * W'` patterns. They remain correct under 0.2.0; not rewritten because the workarounds make the type story explicit, which is pedagogically useful.

### ✅ `M([3, 1, 2], :)` row gather — **landed in rustlab 0.2.0 and 0.3.0**
**Was needed for:** Lesson 11 (FFN per-position independence check) and any later lesson that wants to permute, gather, or sample a subset of rows.
**Now works:** `M([3, 1, 2], :)` returns a matrix of those rows (0.2.0), and `M([3, 1, 2])` returns a *column-major linear gather* (0.3.0; new column-major linear-index semantics — see breaking change below). For ordered row gathers, prefer `M(rows, :)` for clarity.
**Example:**
```
H_perm = H([3, 1, 2, 5, 4], :);   % gather those rows in any order
```

### ✅ Multi-output function definitions — **landed in rustlab 0.3.0** (commit `18523a3`)
**Was hit while writing:** Lesson 18 (training loop) and Lesson 20 (perplexity curve).
**Now works:** `function [dE, dW, L] = step_grad(...)` with `[dE, dW, L] = step_grad(curr, nxt, E, W)` at the call site. The struct-return workaround is no longer needed; lessons 18 and 20 have been migrated to the native multi-output form.

### ✅ Logical `&&` and `||` short-circuit — **landed in rustlab 0.3.0** (commit `18523a3`)
**Was hit while writing:** Lesson 19 (BPE merge step).
**Now works:** `if i < L && seq(i + 1) == val` evaluates LHS first and skips RHS when LHS is false, so the last-position OOB read no longer happens. Lesson 19 has been migrated back to the canonical idiom; the nested-`if` workaround is gone.

### ✅ `layernorm(M)` row-wise matrix overload — **landed in rustlab 0.3.0** (commit `18523a3`)
**Was needed for:** Lesson 12 (LayerNorm sublayer) and every later transformer lesson.
**Now works:** `layernorm(M)` returns a matrix of the same shape with each row normalised to mean 0, std 1. Lessons 12, 13, 14 have been migrated from the per-row loop to the matrix overload.
**Example:**
```
H_normed = layernorm(H);                    % shape (T, d_model), per-row mean=0 std=1
```

### `softmax(M[, dim])` row-wise matrix overload — **requested** (`../rustlab/dev/requests/softmax-matrix-rowwise.md`)
**Hit while writing:** Lesson 08 (attention weights), Lesson 13 (transformer block), Lesson 14 (full GPT), Lesson 15 (backprop through attention).
**Workaround in use:** A per-row loop, `for t = 1:T; A(t, :) = softmax(S(t, :)); end`. Correct, but writes the same loop four separate times across the lesson series and slightly misleads about the parallel nature of softmax (every row is independent).
**Wanted:** `softmax(M)` / `softmax(M, dim)` mirroring `layernorm(M[, dim[, eps]])`, default `dim=2` (per-row, ML convention).
**Example (target):**
```
A = softmax(S_masked, 2);                   % per-row softmax on a T × T scores matrix
```

### ⚠️ BREAKING (rustlab 0.3.0): `M(scalar)` is now a linear-index element, not a row
**Hit during the 0.3.0 audit.** Previously `M(2)` returned the second *row* of a matrix; in 0.3.0 it returns the second column-major *linear element* (matches `find(M)`'s 1-based linear indices and is consistent with vector indexing).
**Migration recipe:** anywhere a script meant "row `t` of M", rewrite as `M(t, :)`. The notebooks and scripts in this repo were swept after the 0.3.0 release; the canonical idioms are:
- Row read: `S(t, :)`, `E(curr, :)`, `H(t, :)`, etc.
- Element read: `M(t)` returns a scalar.
- Row write: `M(t) = vec` still assigns row `t` (legacy compat — `M(scalar) = vec` reads as "assign the vector starting at linear index `t * nrows`", which lines up with the row when `t` is a row index and the RHS is a row vector). For clarity, lessons keep `M(t, j) = scalar` for element writes and `M(t) = vec` for row writes.

### `softmax(logits(1))` after a vector × matrix
**Hit while writing:** Lessons 18 and (preventatively) 16, 17.
**Symptom:** The idiom `logits = h * W; p = softmax(logits(1))` mis-fires when `h` is a vector — `h * W` returns a *vector*, so `logits(1)` extracts the first scalar element. Softmax of a scalar yields a 1×1 matrix that breaks downstream `p(j)` indexing.
**Workaround in use:** Call `softmax(h * W)` directly. The vector-valued result indexes correctly with `p(j)`. If a 1×N matrix really is needed, write `reshape(h * W, 1, vocab)`.
