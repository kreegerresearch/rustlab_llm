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

```
lessons/
  NN-topic-name/
    lesson.md          # theory, equations, learning objectives, exercises
    *.r                # rustlab scripts — one script per concept
    outputs/           # generated SVGs — created at runtime, not committed

notebooks/
  NN-topic-name.md     # rustlab-notebook files — integrated prose, math, and code
  outputs/             # rendered HTML — created at runtime, not committed

PLAN.md                # phase status, handoff notes, acceptance criteria
README.md              # project overview and lesson roadmap
Makefile               # build targets for scripts, notebooks, and cleanup
```

---

## Running

All commands run from the project root:

```bash
make                    # show help
make all                # run all scripts + render notebook site
make scripts            # run all .r scripts
make notebooks          # render notebooks to a site (HTML + auto-generated index.html)
make lesson-01          # run only lesson 01's scripts (works for 01–06)
make notebook-03        # render only lesson 03's notebook
make clean              # delete all generated SVGs and notebook HTML
make clean-scripts      # delete only script SVG outputs
make clean-notebooks    # delete only notebook HTML outputs
```

The notebook render uses directory mode: `rustlab notebook render notebooks/ -o notebooks/outputs` produces one HTML per `.md` plus an `index.html` landing page. `notebooks/index.md` supplies the landing-page title and intro; per-lesson `.md` files carry YAML frontmatter with `title:` and `order:` so the index lists them in lesson order.

Single script:
```bash
rustlab run lessons/01-tokens-and-encoding/char_frequencies.r
```

Interactive REPL:
```bash
rustlab
```

---

## Lesson Format (`lesson.md`)

Use GitHub-flavored Markdown with LaTeX math: `$inline$` and `$$block$$`.

**Required sections (in order):**

1. `## Learning Objectives` — 3–5 bullet points stating what the student will be able to do
2. `## Background` — prerequisite knowledge assumed for this specific lesson
3. `## Theory` — derivations and key equations; derive step-by-step, name every variable, state units
4. `## Core Concepts` — the central idea expressed plainly in 1–2 paragraphs
5. `## Simulations` — for each `.r` file: what it computes, what to observe, what to verify by hand
6. `## Exercises` — 3–5 follow-up questions or script modifications

**Style:**
- Derive equations step-by-step; never skip a step without explanation
- Name every variable and state units explicitly
- Connect math to intuition: explain *why* the result looks the way it does
- Call out common misconceptions explicitly

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
- All plot output goes to `outputs/` relative to the script's directory
- Always `print()` key numerical results a student should verify by hand
- Name files descriptively: `gradient_descent.r`, not `script1.r`
- Keep scripts short enough to read in one sitting (split if over ~60 lines)
- Each script must run independently (no shared state between scripts)

---

## Notebook Conventions (`.md` notebooks)

Notebooks use the `rustlab-notebook` format: standard Markdown files where fenced code blocks tagged `` ```rustlab `` are executed. See `../rustlab/docs/notebooks.md` for the full spec.

**Rules:**
- One notebook per lesson, named to match: `NN-topic-name.md`
- Weave prose and math with executable code blocks in a linear narrative
- Variables persist across code blocks — define once, use in later blocks
- Use LaTeX math in prose: `$inline$` and `$$display$$`
- Use `<!-- hide -->` before setup-only code blocks the reader doesn't need to see
- Use template interpolation `${expr}` to embed computed values in prose (e.g. `${mean(v):%.3f}`)
- Save plots to `outputs/` (same commands as scripts); inline Plotly plots are automatic for `figure()`/`savefig()` calls
- Each notebook should be self-contained — a reader should not need to run the separate `.r` scripts first
- Keep the narrative concise; the `lesson.md` has the full theory and exercises

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

All previously requested functions have been implemented. No outstanding recommendations.
