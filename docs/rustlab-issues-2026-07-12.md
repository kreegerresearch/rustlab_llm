# Rustlab issues found during the 2026-07-12 curriculum audit

Environment: `rustlab` / `rustlab-notebook` **0.3.6**, binaries dated **2026-07-11**, macOS (darwin 25.5.0). Found while auditing all 24 lessons of `rustlab_llm` (54 scripts + 24 notebooks, every script run, full book re-rendered). Ordered by severity. Each item has a minimal repro; workarounds now applied in this repo are noted so they can be reverted when fixed upstream.

---

## Bugs

### 1. BREAKING: `cache` became a reserved word — previously-valid scripts fail to parse

The 0.3.6 `cache` *statement* (`cache enable`, `cache off`, …) reserves the lowercase identifier `cache` in the grammar. Any script using `cache` as a variable name — a very natural name in numerical code (forward-pass caches, memo tables) — no longer parses.

```
cache = 5;                      → error: parse error at line 1: cache: unexpected token Eq
function [y, cache] = f(x) ...  → error: expected identifier in function output list, got Cache
```

Case-sensitive: `Cache = 5;` still works (the error message prints the token's debug name "Cache", which misleads — the reserved word is lowercase `cache`).

**Impact:** four curriculum scripts (`capstone.rlab`, `full_backprop.rlab`, `sft.rlab`, `dpo.rlab`) that ran under the June 0.3.6 build failed to parse under the July build. `make lesson-22` / `lesson-24` were broken at HEAD.

**Suggested fix:** parse `cache` context-sensitively — treat it as a statement keyword only when followed by a known subcommand (`enable`, `off`, `add`, `remove`, `status`, `clear`, `prune`, `list`) or a string literal, and as an ordinary identifier otherwise (assignment, output lists, expressions). MATLAB solves the same ambiguity for command syntax with exactly this rule.

**Workaround in this repo:** variable renamed `cache` → `acts` in the four scripts, with `# TODO` markers to revert.

### 2. `subplot` + `heatmap`/`imagesc`: SVG export renders only the first panel

```
figure();
subplot(1, 2, 1); heatmap({"a","b"}, {"a","b"}, A, "panel one", "viridis");
subplot(1, 2, 2); heatmap({"a","b"}, {"a","b"}, B, "panel two", "viridis");
savefig("two_panels.svg")       % SVG contains only "panel one"
```

Panels 2..n are **silently dropped** from the exported SVG (same with `imagesc`). Line plots and histograms in subplots export all panels correctly. The notebook renderer's captured figures are affected identically.

**Impact:** Lesson 09's central figure — "four attention heads side-by-side" — shipped in the published book with only Head 1 visible while the prose interpreted all four patterns. Any heatmap-grid figure is affected.

**Workaround in this repo:** multi-panel heatmap figures split into consecutive single-heatmap figures, with TODO markers to recombine.

### 3. Unsuppressed `figure()` echoes its integer handle into output

`figure()` at statement position (no semicolon) echoes the new figure's handle. In notebook renders the echo is captured as a bare-integer text block above every plot; in shell runs it prints stray `2`/`3` lines between real outputs. MATLAB's `figure` does not echo.

Two aggravations:

- **Cross-notebook counter churn.** In directory-mode rendering the handle increments across the whole render (lesson 01 echoes `2`, lesson 16 echoes `36`, …), so adding one plot to an early lesson changes the echoed integer in *every later lesson's* rendered output — spurious diffs across the whole book.
- One reviewer observed the echo behaving inconsistently across repeated identical `rustlab run` invocations (echo on first run, absent on repeats). Unconfirmed/racy; worth checking whether the result cache interacts with statement-echo.

**Impact:** before this audit, every one of the ~40 plot blocks in the published book carried a meaningless captured integer that students would try to interpret.

**Suggested fix:** don't echo the handle for a statement-position `figure()` call (match MATLAB), or have the notebook renderer suppress figure-handle echoes.

**Workaround in this repo:** all `figure()` calls converted to `figure();`; authoring conventions updated.

### 4. Unsuppressed `histogram()` echoes its 2×n bin matrix

Same class as #3: `histogram(v)` without `;` dumps its bin-edges/counts return matrix (`Matrix(2x10) ...`) into captured notebook output. Suppressed with `;` in this repo.

### 5. Silent behaviour changes shipped under the same version number

The 2026-07-11 rebuild still reports `0.3.6` but changed observable behaviour vs. the June 0.3.6 build:

- the `cache` reservation (#1) — a parse-level breaking change;
- floating-point last-digit changes in reductions (e.g. `Var(q·k)` sums in lesson 08, `max` in lesson 10 — consistent with a summation-order change);
- heatmap rendering of non-finite data changed: NaN cells now render gray and the colorbar excludes non-finite values (previously NaN/inf leaked into the scale — the new behaviour is *better*, but it changed committed SVGs).

**Impact:** this repo's CI drift guard (`make notebooks-check`) fails on a clean tree because "same version" no longer implies "same output". Version-pin-based workflows can't function.

**Suggested fix:** bump the version (or at least a build/patch identifier surfaced in `--version` and the run banner) on any observable behaviour change.

### 6. `heatmap`/`imagesc` color-map by ABSOLUTE VALUE — negative data renders wrong

The color normalization uses `|v|`, not `v`. Repro (verified on both `heatmap` and `imagesc`, SVG rect fills inspected):

```
B = [-2, -1; 1, 3];
figure(); imagesc(B, "viridis"); savefig("b.svg")
% −2 renders MID-scale (|−2|=2), −1 and +1 render identically at MIN, +3 at MAX.
% B = [-1.5, -0.5; 0.5, 1.5] → ±1.5 both MAX, ±0.5 both MIN.
% B = [5, -5; -5, -5]      → every cell the identical color.
```

**Impact:** any signed matrix plotted as a heatmap is silently wrong — sign structure is discarded and large-magnitude negatives render as *hot*. In this curriculum it made lesson 15's gradient-flow heatmap (log₁₀ norms, negative for sub-1 norms) show the vanishing row as brightest — the plot contradicted the lesson's own prose. The classic sinusoidal positional-encoding heatmap (values in [−1, 1]) renders as `|sin|`/`|cos|`, visually doubling the frequency.

**Suggested fix:** normalize on the signed min/max like MATLAB's `imagesc`/`caxis`. A diverging default palette for mixed-sign data would be a bonus.

**Workaround in this repo:** shift plotted data to be non-negative (e.g. `log10(G / g_min)`, `(PE + 1) / 2`) with an in-code comment, or plot explicitly-labeled magnitudes.

### 7. Single-output `svd` returns `U`, not the singular values

`s = svd(W)` binds the first tuple element (`U`, an m×m matrix) rather than the singular-value vector. MATLAB's single-output convention (`s = svd(A)` → vector of singular values) is what numerically-literate users expect; the current behaviour is a silent footgun.

### 8. Plot argument validation is silent

- `hline(y, "gray", ...)` and `hline(y, "dashed", ...)`: unrecognized color strings silently fall back to a palette default (a "gray" request rendered green; a "dashed" style-in-color-slot mistake rendered solid). A warning on unrecognized color names would surface both bugs.
- `heatmap`/`imagesc` accept NaN/inf data and `bar` accepts all-zero data without any diagnostic. Two published curriculum figures were broken data rendered silently (a NaN probability matrix; an all-zero bar chart). A one-line stderr warning on non-finite plot data would have caught both at render time.

---

## Feature requests

### 9. Lint/diagnostic for scalar linear-indexing of matrices — `M(i)` where a row was meant

The 0.3.0 breaking change (`M(i)` = column-major element, was row `i`) is documented and intentional, but latent instances of the old idiom were **the single most damaging bug class in this curriculum**: this audit found eight more (lessons 04, 05 ×3, 07 ×2 + script, 08, 10, 15), several published with visibly-absurd output (NaN probability matrices, a "machine epsilon" of 5.85e-02, a backward pass failing its own finite-difference check at 1.25). A `rustlab run --lint` (or default stderr note) flagging scalar linear reads of true matrices — especially patterns like `sum(M(i))` inside a loop over rows — would catch every one of these mechanically.

### 10. Struct-field indexing: `s.M(t, :)`

`acts.M(t, :)` parses as a call to function `M(...)` rather than indexing the struct field. Every backward-pass function in lessons 22/24 must unpack ~18 fields into locals (~20 boilerplate lines × 5 scripts) purely to index them. Grammar support for indexing struct-field values directly would remove the boilerplate.

### 11. Integer-aware display for shape-like outputs

`print(size(X))` renders `[1×2]  8.000000  64.000000` — float formatting plus a shape tag for what is conceptually `[8, 64]`. Long vectors also truncate with `... (N total)`, which garbles pedagogical dumps. An integer display for `size()` results (and/or a raw print mode) would clean up every "check the shape" example.

### 12. Previously filed, still the top structural wants

- **Automatic differentiation** (`grad(f, x)` / reverse-mode tape) — every training lesson hand-codes its backward pass.
- **Module/import system** — the transformer forward/backward library is duplicated verbatim across four scripts.

(Both already recorded in `AGENTS.md` → Rustlab Recommendations with full context.)
