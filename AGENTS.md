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

**Curriculum status:** 24 lessons across 10 phases, all complete. Phases 1–8 (Lessons 01–22) cover the nanoGPT / *Attention Is All You Need* baseline end-to-end. Phase 9 (Lesson 23) covers post-2020 architectural variants — RoPE, RMSNorm, SwiGLU, GQA — as drop-in swaps. Phase 10 (Lesson 24) wires the full analytical backward pass through the Lesson 13 transformer block and uses it for SFT and DPO. The Lesson 22 capstone trains the full architecture end-to-end (PPL → 1.00008 on a context-2 corpus where bigram floors at 1.5).

**Learning goal:** Derive every core LLM algorithm — tokenisation, attention, transformer blocks, training, fine-tuning, and inference — with working code and plots at each step. Follows the architecture of [nanoGPT](https://github.com/karpathy/nanoGPT) through Phase 8 and extends it through Phases 9–10.

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
make validate           # lint rendered markdown via `rustlab-notebook validate` (markdownlint-cli2)
make lesson-01          # run only lesson 01's .rlab scripts (pattern target: lesson-NN for any 01–24)
make clean              # delete the interactive HTML build and .rlab artefacts
```

The notebook render is directory-mode: `rustlab-notebook render notebooks --format markdown --output book` produces `book/<slug>.md` plus `book/plots/<slug>/plot-N.svg` for each lesson. The hand-written `book/README.md` is preserved (the renderer skips files named `README.md` on input). Note: as of rustlab 0.3.4 (May 2026) the renderer is a separate binary `rustlab-notebook` — the old `rustlab notebook ...` subcommand has been removed. `make notebooks` is updated accordingly.

`make validate` shells out to `rustlab-notebook validate -f markdown notebooks`, which re-renders every notebook to a temp dir and pipes it through `markdownlint-cli2` using the project-root `.markdownlint-cli2.jsonc` (mirrors rustlab's own noise floor plus `MD024: { siblings_only: true }` so the curriculum's repeated `### Theory` H3s under distinct H2 parents pass). Install the linter with `npm i -g markdownlint-cli2`; without it, validate reports SKIPPED rather than failing. HTML / LaTeX / PDF validation are opt-in via `-f html|latex|pdf` and require additional linters (`vnu` + JRE, `chktex`, `qpdf`/`pdfinfo`).

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

**ML / Activations:** `softmax(v)` / `softmax(M)` (row-wise, 0.3.3+), `relu(v)`, `gelu(v)`, `layernorm(v)` / `layernorm(M)` (row-wise) / `layernorm(v, eps)`

**Array construction:** `zeros(n)` / `zeros(m,n)`, `ones(n)` / `ones(m,n)`, `eye(n)`, `linspace(a,b,n)`, `logspace(a,b,n)`, `rand()` (0.3.4+, scalar in [0,1)) / `rand(n)`, `randn(n)` / `randn(m,n)`, `randi(imax,n)`, `randi([lo,hi],n)`

**Array inspection:** `len(v)`, `length(v)`, `numel(x)`, `size(x)`

**Matrix ops:** `reshape(A,m,n)`, `repmat(A,m,n)`, `transpose(A)`, `diag(v)` / `diag(M)`, `outer(a,b)`, `kron(A,B)`, `inv(M)`, `linsolve(A,b)`, `det(M)`, `trace(M)`, `rank(M)`, `eig(M)`, `svd(A)`, `expm(M)`, `norm(v)` / `norm(v,p)`, `dot(u,v)`, `cross(u,v)`, `meshgrid(x,y)`, `roots(p)`

**Concatenation:** `horzcat(A,B,...)` / `[A, B]`, `vertcat(A,B,...)` / `[A; B]`

**Structs:** `struct("k",v,...)`, `s.field`, `s.field = val`, `isstruct(x)`, `fieldnames(s)`, `isfield(s,"name")`, `rmfield(s,"name")`

**String arrays:** `{"a","b","c"}`, `sa(i)`, `iscell(x)`, `length(sa)`, `numel(sa)`

**Higher-order:** `arrayfun(f, v)`, `feval("name", args...)`, `parmap(f, indices)` (scalar / vector / matrix-returning lambdas; 0.3.3+)

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

This section is the running record of rustlab feature requests, breaking changes, and idiomatic patterns the curriculum has had to adapt to. Three groups, ordered for triage:

1. **Open feature requests** — wanted but not yet landed.
2. **Required idioms / breaking changes** — current rules new code MUST follow.
3. **Landed (✅)** — historical record, most-recent rustlab version first.

When a needed function is missing from rustlab, record it here with the format:

```
### function_name(args) -> return_type
**Needed for:** Lesson NN — [title]
**Purpose:** [what it computes]
**Example:** result = function_name(arg1, arg2)
```

---

## Open feature requests

### Automatic differentiation — `grad(f, x)` or a reverse-mode tape
**Needed for:** Lessons 15, 22, 24 — backpropagation, the capstone training loop, and full-backprop fine-tuning (SFT/DPO).
**Purpose:** Compute gradients of a scalar loss w.r.t. parameter matrices without hand-deriving and hand-coding every backward op. Today the curriculum derives the chain rule analytically and codes each backward pass by hand (`backward`, `layernorm_bwd`, `gelu_grad`, …). This is pedagogically valuable once (Lesson 15/24) but forces every training script to carry a bespoke, error-prone backward path.
**Current state (0.3.6):** Only numeric grid gradients exist — `gradient` (2-D scalar field) and `gradient3` (3-D). There is no autodiff over the expression graph.
**Example (target):** `[dW, db] = grad(@() loss(W, b, batch), {W, b});`

### Module / import system — share code across `.rlab` scripts
**Needed for:** Lessons 22 and 24, where the full-transformer forward/backward library (`forward`, `backward`, `layernorm_fwd/bwd`, `gelu_grad`) is **duplicated verbatim** across `full_backprop.rlab`, `sft.rlab`, and `dpo.rlab` because rustlab has no way to include shared definitions and AGENTS.md requires self-contained scripts.
**Purpose:** Let a script pull common functions from a shared file so the transformer library lives in one place.
**Current state (0.3.6):** No `import` / `include` / `require`; each script must be self-contained.
**Example (target):** `import "lib/transformer.rlab"` (or similar) at the top of a lesson script.

### ~~CLI should announce itself as the `.rlab` handler~~ — ✅ landed in 0.3.6
Resolved: `rustlab run` now prints a one-line stderr banner identifying the version and file. See the Landed ✅ → rustlab 0.3.6 section below.

---

## Required idioms (breaking changes and rules)

### ⚠️ BREAKING (rustlab 0.3.0): `M(scalar)` is now a linear-index element, not a row
**Hit during the 0.3.0 audit.** Previously `M(2)` returned the second *row* of a matrix; in 0.3.0 it returns the second column-major *linear element* (matches `find(M)`'s 1-based linear indices and is consistent with vector indexing).
**Migration recipe:** anywhere a script meant "row `t` of M", rewrite as `M(t, :)`. The notebooks and scripts in this repo were swept after the 0.3.0 release; the canonical idioms are:
- Row read: `S(t, :)`, `E(curr, :)`, `H(t, :)`, etc.
- Element read: `M(t)` returns a scalar.
- Row write: rustlab 0.3.4 added the symmetric `M(t, :) = vec` form; new code should prefer it. The legacy `M(t) = vec` (assign linear-index starting at `t * nrows`, which lines up with row `t` when the RHS is a row vector) still works and is bit-identical, so existing scripts have been left as-is to avoid churn.

### `softmax(logits(1))` after a vector × matrix
**Hit while writing:** Lessons 18 and (preventatively) 16, 17.
**Symptom:** The idiom `logits = h * W; p = softmax(logits(1))` mis-fires when `h` is a vector — `h * W` returns a *vector*, so `logits(1)` extracts the first scalar element. Softmax of a scalar yields a 1×1 matrix that breaks downstream `p(j)` indexing.
**Workaround in use:** Call `softmax(h * W)` directly. The vector-valued result indexes correctly with `p(j)`. If a 1×N matrix really is needed, write `reshape(h * W, 1, vocab)`.

### ⛔ `break` / `continue` — **declined upstream**; use `while ... && cond` instead
**Status:** The rustlab project has declined to add `break` / `continue` keywords. The canonical idiom for early-exit in this curriculum is a `while` loop whose condition encodes "keep going until the hit", relying on short-circuit `&&` (rustlab 0.3.0+) to guard the bound check.
**Required pattern** for "find the first index that satisfies a predicate":
```
% Walk forward to the first index whose cumulative mass clears P.
j = 1;
while j < K && c(j) < P
  j = j + 1;
end
n_keep = j;
```
**Required pattern** for inverse-CDF sampling (was `for ... return;`):
```
% Walk the cumulative distribution until we cross r.
c = cumsum(p);
r = rand();                      % rustlab 0.3.4+
N = length(p);
i = 1;
while i < N && c(i) < r
  i = i + 1;
end
tok = i;
```
**Do not use** any of these older workarounds in new code:
- `break;` (errors with `undefined variable 'break'`)
- `continue;` (same)
- The `found = 0/1` flag inside a `for` loop
- `return;` from a `for` loop as a mid-loop exit

All currently-committed scripts and notebooks use the `while` form; new lessons should follow suit.

### Scalar-indexing pitfall: `length(scalar)` works, but `scalar(j)` does not
**Symptom:** `x = 5; x(1)` errors with `undefined function 'x'` even though `length(5)` returns 1.
**Rule:** Wrap a scalar in `[id]` (a 1×1 matrix) at any point where downstream code will index into the result with `value(j)`. Concretely, the capstone's `expand_token` returns `[id]` for the terminal base case so that `char_names(chars_i(j))` works in the caller.
**Use the bare scalar** when the consumer is `length(.)`, arithmetic (`x + y`), or concatenation (`[x, y]`).

---

## Landed ✅

Resolved feature requests and fixed bugs, most-recent rustlab version first.

### rustlab 0.3.6

**`rustlab run` self-identifying banner.** Resolves the long-standing "CLI should announce itself as the `.rlab` handler" feature request. `rustlab run foo.rlab` now emits a one-line stderr banner before execution, e.g. `rustlab 0.3.6 — interpreting foo.rlab (.rlab)`, giving every shell-pasteable command clear provenance (rustlab, not MATLAB). No script or notebook changes needed.

**`imagesc` y-axis orientation now matches MATLAB/Octave.** `imagesc` previously rendered matrix row 1 at the top (image convention) but labelled the y-axis bottom-to-top (physics convention) — the two silently disagreed. 0.3.6 aligns both to MATLAB/Octave exactly: image-convention render **and** reversed y-axis labels (row 0 at the top), default `axis("ij")`. New panel controls `axis("xy")` (row 0 at bottom, for physics/meshgrid plots), `axis("ij")` (default), and process-wide `set_default_axis(...)`. **Impact on this curriculum: none** — a full `make notebooks` re-render under 0.3.6 produced byte-identical plot SVGs (the change affects the interactive/plotters path, not the notebook SVG-export backend). All heatmaps (`imagesc(M, "viridis")` in lessons 05–16) are unaffected. Use `axis("xy")` only if a future lesson needs physics-up orientation.

**Markdown renderer strips trailing blank lines.** `rustlab-notebook render --format markdown` now collapses blank-line runs and strips trailing blanks. A 0.3.6 re-render removed exactly one trailing blank line from each `book/*.md` file; no other content changed.

**`rustlab notebook` subcommand removed → standalone `rustlab-notebook`.** Notebook rendering now lives entirely in the separate `rustlab-notebook` binary (`rustlab-notebook render notebooks --format markdown ...`). The `Makefile` already invokes `rustlab-notebook` directly, so the `notebooks` / `html` targets are unaffected.

**Persistent function-result cache (`rustlab cache`).** New `rustlab cache status|list|clear|prune` subcommand inspecting an on-disk cache of function results. Optional performance feature; the curriculum does not rely on it.

### rustlab 0.3.4

**`A(i, :) = vec` symmetric row-write.** `A(i, :) = vec` writes a row exactly symmetric to the `A(i, :)` row-read. The older `A(i) = vec` legacy form still works and produces identical results; new code should prefer the symmetric `A(i, :)` form. Existing scripts continue to use the legacy form — bit-identical, left in place to avoid churn. One latent correctness bug was uncovered along the way: Lesson 10's `X_tok(t) = E_pe(ids(t))` had silently broken under the 0.3.0 M(scalar) breaking change (the RHS returns a scalar, not a row); fixed to `X_tok(t, :) = E_pe(ids(t), :)` in 0.3.4.

**`rand()` zero-arg form.** Returns a scalar in `[0, 1)`, matching MATLAB / Octave convention. The `rand(1)(1)` chain-index workaround is no longer needed. Migrated lessons 21 and 22 `sample_categorical` helpers.

**`length(scalar)` returns 1.** A function that may return a scalar OR a vector composes cleanly with `[L, R]` concatenation and with downstream `length()` calls. **However, scalar indexing still errors** — see "Scalar-indexing pitfall" in the Required idioms section above.

**Strided LHS assignment.** `v(1:2:6) = [1, 2, 3]` writes the strided slice in one assignment. The lessons did not previously rely on it (used element-by-element writes); it is now available for future scripts.

### rustlab 0.3.3

**`softmax(M)` row-wise matrix overload.** `softmax(M)` returns a matrix of the same shape with each row softmax-normalised (dim=2, ML convention). One call replaces the `for t = 1:T; A(t) = softmax(S(t, :)); end` idiom. Migrated lessons 21 (`kv_cache.rlab`) and 23 (`gqa.rlab`) — the two scripts written before 0.3.3 that still had the per-row loop. Earlier lessons 08, 13, 14, 15 already used the matrix overload (migrated on the original feature request).

Example: `A = softmax(S_masked);` — per-row softmax on a T × T scores matrix.

**`parmap` with vector/matrix-returning lambdas.** `parmap(f, 1:N)` where `f(i)` returns a $d$-vector produces an $N \times d$ matrix (row-stacked). Every row-/position-/head-parallel transformer pattern is now expressible as a single `parmap` call. The Lesson 20 sidebar's table was updated to reflect the new capability. Pre-existing per-row `for` loops remain pedagogically explicit and were not rewritten.

Examples that now work:
```
A = parmap(@(t) softmax(S(t, :)), 1:T);                   % per-row softmax → T × T
H_out = parmap(@(t) ffn(H(t, :), W1, b1, W2, b2), 1:T);   % per-position FFN → T × d_model
```

### rustlab 0.3.2

**Renderer math-escape regression fixed.** Rustlab 0.3.1's markdown renderer had doubled every backslash spacing command inside LaTeX math (`\;` → `\\;`, `\!` → `\\!`, `\,` → `\\,`, `\|` → `\\|`) and rewrote `^*` → `^{\ast}`. 0.3.2 restored single-backslash output. `make notebooks` now produces bit-identical output to the pre-0.3.1 renders for every unchanged source file. Lessons no longer need to avoid those constructs in math.

### rustlab 0.3.0

**Multi-output function definitions.** `function [dE, dW, L] = step_grad(...)` with `[dE, dW, L] = step_grad(curr, nxt, E, W)` at the call site. The struct-return workaround is no longer needed; lessons 18 and 20 have been migrated to the native multi-output form.

**Logical `&&` and `||` short-circuit.** `if i < L && seq(i + 1) == val` evaluates LHS first and skips RHS when LHS is false, so the last-position OOB read no longer happens. Lesson 19 uses the canonical idiom; the nested-`if` workaround is gone.

**`layernorm(M)` row-wise matrix overload.** Returns a matrix of the same shape with each row normalised to mean 0, std 1. Lessons 12, 13, 14 use the matrix overload, no per-row loop.

Example: `H_normed = layernorm(H);` — shape (T, d_model), per-row mean=0 std=1.

### rustlab 0.2.0

**Vector + 1×N matrix arithmetic.** `vec + 1×N_matrix`, `vec .* 1×N_matrix`, etc. — implicit broadcasting promotes both sides. The `(W * x')'` returns-a-matrix path no longer breaks per-step updates like `x = x + alpha * f(x)`. Lessons still use the explicit `M(1)` row-extract and `x * W'` patterns in places because the workarounds make the type story pedagogically explicit.

**`M([3, 1, 2], :)` row gather.** Returns a matrix of those rows. `M([3, 1, 2])` returns a column-major linear gather (introduced in 0.3.0). For ordered row gathers, prefer `M(rows, :)` for clarity.

Example: `H_perm = H([3, 1, 2, 5, 4], :);` — gather rows in any order.

### rustlab 0.1.x

**`seed(n)`.** `seed(N)` sets the global RNG to a deterministic state; subsequent `rand` / `randn` / `randi` / `sprand` calls are bit-stable. `seed()` (no argument) re-randomises from system entropy. Lessons 04 and 05 originally used a sin/cos pseudo-random matrix and a hand-set `draws` vector with TODO markers; both have been updated to use `seed(N)` followed by `randn` / `rand`.

Example: `seed(42); E = randn(8, 6) * 0.1;` — bit-identical across runs.
