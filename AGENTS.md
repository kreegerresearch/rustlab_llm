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

**Curriculum status:** 27 notebooks (Lessons 00–26), all complete. Lesson 00 is the course map. Phases 1–8 (Lessons 01–23) cover the nanoGPT / *Attention Is All You Need* baseline end-to-end: Lesson 22 derives the full analytical backward pass through the Lesson 13 block (the shared `lib/transformer.rlab`), and the Lesson 23 capstone trains the full architecture end-to-end (PPL → 1.00008 on a corpus whose optimal-bigram floor is ≈ 1.47). Phase 9 (Lesson 24) covers RoPE, RMSNorm, SwiGLU, GQA; Phase 10 (Lesson 25) SFT and DPO; Lesson 26 quantisation and fixed-point inference. Phases 11–15 (2026-09-27) moved the toolchain to rustlab 0.3.7, introduced `lib/`, renumbered 22–25, and re-centred every lesson on the **Engineering Lenses** (Signals / Systems / Information, see below) per `docs/proposal-2026-09-27-ee-controls-it-revision.md`; the record is in `PLAN.md`.

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

lib/
  transformer.rlab       # causal_mask, sinusoidal_pe, mha_block_forward (L13/14),
                         # transformer_forward/backward + AdamW + schedule (L22/23/25)
  sampling.rlab          # sample_categorical, topk_mass, topp_mass (L21/23)
  bigram_lm.rlab         # train_bigram_lm — the Lesson 18 model used by L20/21
  info.rlab              # entropy_bits/nats, cross_entropy_bits, kl_bits, row_entropies_bits, ppl_from_nats

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
make lesson-01          # run only lesson 01's .rlab scripts (pattern target: lesson-NN for any 01–26; fails on the first script error)
make clean              # delete the interactive HTML build and .rlab artefacts
```

The notebook render is directory-mode: `rustlab-notebook render notebooks --format markdown --output book --jail-root .` produces `book/<slug>.md` plus `book/plots/<slug>/plot-N.svg` for each lesson. The `--jail-root .` is required so notebook blocks may `run "../lib/<name>.rlab"` (0.3.7 jails notebook file I/O to the collection root by default). The hand-written `book/README.md` is preserved (the renderer skips files named `README.md` on input). Note: as of rustlab 0.3.4 (May 2026) the renderer is a separate binary `rustlab-notebook` — the old `rustlab notebook ...` subcommand has been removed. `make notebooks` is updated accordingly.

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
- **Shared code:** pull a library in with a hidden block placed before first use — `<!-- hide -->` then a ```` ```rustlab ```` block containing `run "../lib/transformer.rlab"` (quoted path). Do **not** `run` a lesson script from a notebook: a `run` nested inside that script resolves relative to the *notebook* directory and escapes the jail. Notebooks carry their own short, library-based code blocks that parallel the scripts.
- **One captured plot per code block** in markdown output. Side-by-side panels must be built with `subplot(rows, cols, idx)` inside a single `figure();` in one block (heatmap subplots export correctly since 0.3.7). `<!-- grid: N -->` affects HTML only.
- Heatmaps colour by signed value (0.3.7) — plot signed matrices directly; never shift them non-negative.
- The renderer captures the active figure automatically. **Don't** call `savefig()` inside notebook code blocks.
- Use `figure();` (not `clf;`) at the start of each plot block — **with the semicolon**: an unsuppressed `figure()` echoes its integer handle into the captured output, and the handle increments across the whole directory render, so one bare `figure()` puts a meaningless (and render-order-dependent) number in the book. Same rule for `histogram(...);`, which otherwise echoes its 2×n bin matrix.
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

## Engineering Lenses (the course's organising device, Phases 12–15)

The course teaches LLM design from **signals and systems**, **modern control / dynamical systems**, and **information theory**. Every lesson 01–26 therefore closes its concept content with one fixed H2, placed before `## Key Takeaways`:

```
## Engineering Lenses
### Signals
### Systems
### Information
```

Rules:

- **Each H3 answers a fixed question.** *Signals*: what is the signal (its axis, units, scale) and what operation acts on it (filter, transform, modulation, normalisation) — with a frequency-domain view when that is meaningful. *Systems*: what is the state, what is the update law, is it stable and what sets its time constant or damping, where is the feedback. *Information*: what bits are created, moved, or destroyed here; what is the floor, bound, or budget — **computed on the lesson's own data**.
- **Every claim opens with a bold honesty label**: `**Exact.**` (a formal equivalence), `**Model.**` (a faithful simplification), or `**Analogy.**` (intuition only, not an identity). Never let an analogy read as an identity.
- **Every H3 that is present contains at least one executed rustlab block** (a computation or a figure). Prose-only lenses are not allowed. A lens with nothing exact or model-grade to say is omitted, and the H2 carries one sentence saying so (e.g. "No signals reading adds to this lesson.").
- The former `## Connection to Information Theory` sections migrate into `### Information`, converted from assertion to computation.
- **Budget:** the lenses add at most one H2 per lesson (typically 40–120 lines); anything larger becomes an exercise or a forward reference. A rewritten lesson should not grow by more than about 40 % in lines.
- **Figures:** under every figure, one `> [!TIP]` line saying what to look for. Side-by-side panels use `subplot` in one block (one captured plot per block). Signed data is plotted signed.
- **Diagrams:** architectural lessons carry a ```` ```mermaid ```` signal-flow / block diagram with tensor shapes on the edges. Use `flowchart LR` or `TD`, short node labels, and quote any label containing parentheses, `|`, or `#` (`A["H (T × d)"]`). The markdown renderer passes the fence through to GitHub; the HTML renderer draws it offline.
- **Demonstrations must demonstrate:** a hand-built "interpretable" matrix is printed and the prose names the entry that proves the claim; a figure described in prose exists in the notebook.
- **Shared helpers:** `lib/info.rlab` (`entropy_bits`, `entropy_nats`, `cross_entropy_bits`, `kl_bits`, `row_entropies_bits`, `ppl_from_nats`) — include with the hidden `run "../lib/info.rlab"` block once the concept has been derived in that lesson (Lesson 02 derives entropy inline first).

### Notation (fixed in Lesson 00 — every lesson follows it)

| Symbol | Meaning |
|---|---|
| $T$ | sequence length; tokens index discrete time $t = 1, \dots, T$ and are the **rows** of every token matrix |
| $d$, $d_{\text{model}}$, $d_k$, $d_{\text{ff}}$ | feature widths; features are the **columns** (channels) |
| $\mathbf{X}, \mathbf{H} \in \mathbb{R}^{T \times d}$ | token / residual-stream matrix; row $t$ is token $t$; layers act by right-multiplication $\mathbf{X}\mathbf{W}$ with $\mathbf{W} \in \mathbb{R}^{d_{\text{in}} \times d_{\text{out}}}$ — math and code agree |
| $\mathbf{W}_Q, \mathbf{W}_K, \mathbf{W}_V, \mathbf{W}_O, \mathbf{W}_1, \mathbf{W}_2, \mathbf{W}_U, \mathbf{E}$ | learned projections, FFN weights, LM head, embedding |
| $\mathbf{A}$ | attention / mixing matrix: rows = query time $t$, columns = key time $i \le t$ (Lesson 07's uniform version is the special case $\mathbf{A} = \mathbf{W}_{\text{avg}}$) |
| $\tau$ | softmax temperature — never $T$ |
| $\mathbf{P}$ | bigram transition matrix (row-stochastic); $\boldsymbol{\Pi}$ a permutation matrix |
| $\mathcal{L}$ | loss, computed in **nats** in code; prose may report **bits** ($\div \ln 2$) when labelled; perplexity $= e^{\mathcal{L}_{\text{nats}}} = 2^{\mathcal{L}_{\text{bits}}}$ — state the base once per lesson |
| $\bar{\mathbf{x}} = \partial \mathcal{L} / \partial \mathbf{x}$ | adjoint (= the costate $\boldsymbol{\lambda}$ of optimal control; Lesson 15 says so once) |
| $\eta$, $\beta_1$, $\beta_2$, $\mu$ | learning rate, Adam moment decays, heavy-ball momentum |
| indices | 1-based in prose, tables, and code; categorical plot axes carry 1-based labels |

### Exercise solutions

Exercises stay plain numbered lists. For derivation-type exercises only, a hand-written HTML block follows the item (GitHub renders it collapsed):

```
<details><summary>Solution</summary>

…markdown…

</details>
```

Do not use the `<!-- solution -->` directive — it is broken in markdown output (rustlab roadmap A9).

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
- Each script must run independently given `lib/`: shared code is pulled in with `run "../../lib/<name>.rlab"` (quoted, script-relative path) at the top of the script; never copy a function between scripts

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
- Include another file: `run "path.rlab"` (quote the path — an unquoted leading `..` fails to lex); resolves relative to the calling script, or to the notebook for notebook blocks

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

**Canonical save pattern.** The shorthand `savebar`, `savescatter`, `saveimagesc`, and `savehist` wrappers are deprecated — use the `plot/bar/scatter/imagesc` call followed by `savefig(file)`. Note the semicolon on `figure();` (see the authoring rules — bare `figure()` echoes its handle):

```
figure();
bar(y, "title")                    % or: scatter(x, y, "title")
savefig("outputs/chart.svg")

figure();
imagesc(M, "viridis")
title("Heatmap")
savefig("outputs/heatmap.svg")
```

**Multi-panel heatmaps and signed data (fixed in 0.3.7):** `subplot` + `heatmap`/`imagesc` SVG export renders every panel, and heatmaps colour by signed value. The 0.3.6-era split figures and non-negative shifts were removed in Phase 11; do not reintroduce them.

---

## Rustlab Recommendations

This section is the running record of rustlab feature requests, breaking changes, and idiomatic patterns the curriculum has had to adapt to. A standalone, upstream-facing bug report from the 2026-07-12 full-curriculum audit (repros, impact, suggested fixes) lives at `docs/rustlab-issues-2026-07-12.md`. Three groups, ordered for triage:

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

### Struct-field indexing — `s.M(t, :)`
**Needed for:** Lessons 22, 24 — every backward-pass function.
**Purpose:** Index a matrix stored in a struct field directly. Today `acts.M(t, :)` parses as a call to a function `M(...)`, so each backward function unpacks ~18 cache fields into locals (~20 boilerplate lines × 5 scripts).
**Current state (0.3.7):** Still not supported (`s.M(2, :)` → `undefined function 'M'`); also cannot destructure into fields (`[P.E, M.E] = f(...)`). `lib/transformer.rlab` unpacks cache/parameter fields to locals and uses temporaries in `adamw_step`.
**Example (target):** `Q_row = cache.Q(t, :);`

### Lint for scalar linear-indexing of matrices — `M(i)` where a row was meant
**Needed for:** the whole curriculum. The 2026-07-12 audit found **eight** latent instances of the pre-0.3.0 row idiom (lessons 04, 05 ×3, 07 ×2 + script, 08, 10, 15), several published with visibly-broken output (NaN probability matrices, a backward pass failing its own finite-difference check). This is the single most damaging bug class the curriculum has hit.
**Purpose:** `rustlab run --lint` (or a default stderr note) flagging scalar linear reads of true matrices, especially `sum(M(i))` / `p = P(i)` patterns inside loops over rows.

### Integer `size()` display
**Purpose:** `print(size(X))` renders `[1×2] 8.000000 64.000000` for what is conceptually `[8, 64]`. (The single-output `svd` half of this request landed in 0.3.7.)

### `A^k` on a square matrix is element-wise
**Needed for:** the Phase 13 controls lessons (powers of `I − ηH`, Markov-chain `P^t`).
**Current state (0.3.7):** `[0,1;-1,-0.5]^2 ≠ A*A` — silent wrong answers (rustlab roadmap A8). Write `A*A`, a loop, or `expm`.

### `<!-- solution -->` directive breaks markdown output
**Current state (0.3.7):** emits an unclosed and duplicated `<details>` in `-f markdown` (rustlab roadmap A9). Exercises stay plain numbered lists; use hand-written `<details><summary>` HTML only for derivation-type solutions until fixed.

### Plotting and language quirks found during the 2026-09-27 rewrite (0.3.7)
- `hline`/`yline` called after `plot`/`bar`/`semilogy` **replaces** the series unless `hold("on")` is active — always `hold("on")` before adding reference lines (this had blanked `gqa.rlab`'s curve).
- The SVG backend drops `plot`/`scatter` series overlaid on `contour`/`contourf`/`imagesc` (all call orders); draw level sets as `plot` lines or use two figures.
- `quiver` needs gridded origins and rescales shaft lengths to the grid; one or two arrows at a point render wrong. Arrows are drawn as short `plot` polylines.
- `hold("on")` must be re-issued after each `subplot(...)`.
- `svd` has an absolute floor ≈ 1e-8 (`svd(diag([1, 1e-9, 1e-12]))` → `[1, 0, 0]`); do not report σ_min below it.
- `V(:, i)'` does not transpose a column slice (it stays a vector); `M(:, j)'` likewise; use `reshape` or `dot`. `diag(D)` returns a row.
- Indexed compound assignment `C(i, j) += 1` does not parse — write it out.
- Integer-class matrix `*` is element-wise (`int32(A) * int32(B)` = `A .* B`).
- `snr` and `histogram` reject 1×n matrices — pass `randn(n)` vectors or `reshape`.
- `print` accepts at most 16 arguments; there is no `vline` (draw a two-point series).
- Template interpolation: `${expr}` immediately followed by `\times$` or `> 1$`, and the `$${expr}$$` form, are not expanded — put a space or reword.
- Wikilinks inside `> [!NOTE]`/`[!TIP]` callouts are **not** transformed by the markdown renderer — use plain `[text](NN-slug.md)` links there.

### `run` path handling
**Current state (0.3.7):** an unquoted `run ../x.rlab` fails to lex (`invalid number: ..`); the quoted form works. A `run` nested inside a script that a notebook runs resolves relative to the notebook directory, not the script. Both are worked around by convention (quoted paths; notebooks never `run` lesson scripts).

### Markdown render captures one plot per code block
**Current state (0.3.7):** a block with several `figure()` calls emits one SVG in `-f markdown`; `<!-- grid: N -->` is HTML-only. Side-by-side panels use `subplot`.

### ~~Module / import system~~ — ✅ resolved by `run` (0.3.7, verified 2026-09-27)
`run "file.rlab"` merges a file's functions and variables into the caller's scope, from scripts and from notebook blocks (with `--jail-root .`). `lib/transformer.rlab`, `lib/sampling.rlab`, `lib/bigram_lm.rlab` replaced ≈ 700 duplicated lines in Phase 11.

### ~~Warning on non-finite / degenerate plot data~~ — ✅ landed in 0.3.7
### ~~MATLAB-convention single-output `svd`~~ — ✅ landed in 0.3.7
### ~~CLI should announce itself as the `.rlab` handler~~ — ✅ landed in 0.3.6

---

## Required idioms (breaking changes and rules)

### ~~`cache` is a reserved word~~ — ✅ fixed in 0.3.7 (soft keyword)
The 0.3.6 `cache` statement had reserved the lowercase identifier; 0.3.7 made it a soft keyword. `lib/transformer.rlab` uses `cache` for the activation cache again; the `acts` rename and its `# TODO` markers were removed in Phase 11.

### ⚠️ Same-version behaviour drift: the 2026-07-11 binary still reports 0.3.6
The July rebuild changed observable behaviour without a version bump: the `cache` reservation above, floating-point last-digit changes in reductions, and heatmap NaN handling (NaN cells now gray, colorbar excludes non-finite). `make notebooks-check` diffs against books rendered by the June build will show those deltas. When the drift guard fires with no source change, suspect a binary update — check the binary's mtime — and re-render. Details: `docs/rustlab-issues-2026-07-12.md` §5.

### ⚠️ BREAKING (rustlab 0.3.0): `M(scalar)` is now a linear-index element, not a row
**Hit during the 0.3.0 audit.** Previously `M(2)` returned the second *row* of a matrix; in 0.3.0 it returns the second column-major *linear element* (matches `find(M)`'s 1-based linear indices and is consistent with vector indexing).
**Migration recipe:** anywhere a script meant "row `t` of M", rewrite as `M(t, :)`. The notebooks and scripts in this repo were swept after the 0.3.0 release — but the sweep was incomplete: the 2026-07-12 audit found and fixed eight more latent instances (lessons 04, 05, 07, 08, 10, 15), several of which had shipped visibly-broken rendered output. Treat any remaining `sum(M(i))` / `p = P(i)` inside a row loop as a bug until proven otherwise. The canonical idioms are:
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

### rustlab 0.3.7 (verified 2026-09-27; binary built from `../rustlab` main @ `edd2d64`)

**Curriculum impact (Phase 11):** `cache` soft keyword; heatmaps colour by signed value; `subplot` + heatmap SVG exports all panels; bare `figure()`/`histogram()` no longer echo; single-output `svd` returns singular values; `rustlab run` exits 1 on error (so `make lesson-NN` gates CI); stderr warnings for NaN/Inf plot data and unrecognised colours (`"gray"`/hex now accepted); `run "file.rlab"` include (see the module-system entry above); column-range slicing `Q(:, a:b)` and region writes; elementwise two-argument `max`/`min`; `tic`/`toc`; integer types `int8…uint64` and `qfmt`/`quantize`/`snr` (basis for the planned Lesson 26); native complex phasors for RoPE/PE; controls and DSP toolboxes (`eig`, `expm`, `lyap`, `tf`, `bode`, `step`, `freqz`, `fft`) for Phases 12–14; `frame()` + `saveanim("x.gif")` captured into the markdown book; mermaid fences rendered offline and passed through to GitHub. Notebook I/O is jailed to the collection root — the Makefile passes `--jail-root .`. A full re-render under 0.3.7 was byte-identical to the 0.3.6 book except one lesson-04 legend label (labelled-scatter fix).

### rustlab 0.3.6

**`rustlab run` self-identifying banner.** Resolves the long-standing "CLI should announce itself as the `.rlab` handler" feature request. `rustlab run foo.rlab` now emits a one-line stderr banner before execution, e.g. `rustlab 0.3.6 — interpreting foo.rlab (.rlab)`, giving every shell-pasteable command clear provenance (rustlab, not MATLAB). No script or notebook changes needed.

**`imagesc` y-axis orientation now matches MATLAB/Octave.** `imagesc` previously rendered matrix row 1 at the top (image convention) but labelled the y-axis bottom-to-top (physics convention) — the two silently disagreed. 0.3.6 aligns both to MATLAB/Octave exactly: image-convention render **and** reversed y-axis labels (row 0 at the top), default `axis("ij")`. New panel controls `axis("xy")` (row 0 at bottom, for physics/meshgrid plots), `axis("ij")` (default), and process-wide `set_default_axis(...)`. **Impact on this curriculum: none** — a full `make notebooks` re-render under 0.3.6 produced byte-identical plot SVGs (the change affects the interactive/plotters path, not the notebook SVG-export backend). All heatmaps (`imagesc(M, "viridis")` in lessons 05–16) are unaffected. Use `axis("xy")` only if a future lesson needs physics-up orientation. *(Caveat added 2026-07-12: the "byte-identical" claim held for the June 0.3.6 build; the 2026-07-11 rebuild — still reporting 0.3.6 — changed heatmap NaN rendering and reduction FP digits. See the "Same-version behaviour drift" idiom above.)*

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
