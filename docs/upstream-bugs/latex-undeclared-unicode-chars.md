# Bug: LaTeX renderer aborts on undeclared Unicode characters used in LLM content

> [!NOTE]
> **Status (2026-05-23): fix in flight upstream.** When this report was drafted,
> the rustlab repo already had branch `feature/latex-unicode-coverage` (commit
> `e46b393`, "LaTeX preamble: declare 7+ Unicode codepoints surfaced by
> rustlab_llm") covering exactly the 7 characters identified below plus the
> natural sibling sets (full digit sub/superscripts, `↗ ↖ ↙`). No separate
> upstream issue was filed — this document is kept as the original validation
> record and as the bottom-line acceptance test for the upstream branch.

**Reported from:** `rustlab_llm` curriculum, validated against `rustlab-notebook` 0.3.6.
**Affected binary:** `rustlab-notebook render --format pdf` (LaTeX → pdflatex pipeline).
**Reproduced via:** `rustlab-notebook validate -f pdf notebooks/` on the rustlab_llm tree, 2026-05-23.

## Problem

`rustlab-notebook` 0.3.6 ships a `\newunicodechar{…}{…}` declaration block (see `crates/rustlab-notebook/src/render_latex.rs:200-265`) that covers ~50 commonly-used math symbols, Greek letters, arrows, superscripts, and box-drawing characters. Several characters that appear naturally in ML / LLM curriculum prose are **not** in the block, and pdflatex aborts on each one with a fatal `! LaTeX Error: Unicode character …` — no PDF is produced.

Running `rustlab-notebook validate -f pdf notebooks/` on the rustlab_llm tree (24 lessons): **16 PASS / 8 FAIL**, each FAIL traced to one undeclared character.

| Char | Codepoint | LaTeX target | Lesson source |
|---|---|---|---|
| `η` | U+03B7 | `\ensuremath{\eta}` | `06-linear-layers-and-gradient-descent.md:126` |
| `̄` (combining macron) | U+0304 | `\bar{}` (text mode) | `07-context-and-naive-averaging.md:165` (`X̄`) |
| `┬` (box-drawing T) | U+252C | `+` (matches existing `─ │ └ ├` mapping) | `13-transformer-block.md:31,33` (ASCII transformer diagram) |
| `₁` | U+2081 | `\ensuremath{_1}` | `16-adamw-optimizer.md:224,225` (`θ₁`) |
| `↘` | U+2198 | `\ensuremath{\searrow}` | `18-training-loop.md:214` |
| `⁵` | U+2075 | `\ensuremath{^5}` | `19-byte-pair-encoding.md:26,27` (`10⁵`) |
| `✓` | U+2713 | `\ensuremath{\checkmark}` (needs `\usepackage{amssymb}`) | `24-full-backprop-and-fine-tuning.md:32` |

Every one is a single fatal error per build log; the build aborts before reaching downstream lessons.

## Minimal reproducer

```bash
# In a fresh repo with one notebook:
mkdir nb && cat > nb/eta.md <<'EOF'
# Eta

Learning rate $\eta = 0.05$.
EOF

rustlab-notebook render nb --format pdf --output /tmp/out
# → ! LaTeX Error: Unicode character η (U+03B7)
#   Fatal error occurred, no output PDF file produced!
```

Same shape for the other six characters; swap `η` for each.

## Why these specifically

- `η` (eta) is the canonical symbol for **learning rate** across the entire ML literature. It will appear in any curriculum that touches gradient descent.
- `θ₁`, `θ₂`, … with subscript digits are the canonical notation for parameter components — used in any explanation of loss surfaces, Adam, etc. The subscript and superscript Unicode forms (`₁ ₂ ₃ ⁵`) are common in tables and inline prose where `$_1$` is too noisy.
- `↘` and `↗` (and `↑ ↓`) are used in plain-text descriptions of loss curves / overfit diagnostics.
- `✓` and `✗` are used in markdown tables to mark correct vs. incorrect outcomes.
- Box-drawing `┬` joins the `─ │ └ ├` family already in the block — looks like a simple omission rather than a deliberate exclusion.
- The combining macron is the trickiest of the seven: characters like `X̄` are decomposable (`X` + `̄`). `\newunicodechar` on the standalone combining mark would map it to `\bar{}`-style logic; consider whether to handle the precomposed forms instead or to ASCII-fold the combining mark.

## Suggested fix

Extend the existing block in `crates/rustlab-notebook/src/render_latex.rs:200-265` with the seven mappings above. `↗ ↑ ↓` and the remaining subscript/superscript digits (`₀ ₂ ₃ ₄ ₅ … ⁰ ¹ ⁴ ⁶ ⁷ ⁸ ⁹`) and `✗` are likely to surface next as the curriculum grows; consider adding the full set in one pass:

```text
% Greek (additions)
\newunicodechar{η}{\ensuremath{\eta}}

% Subscripts (digits)
\newunicodechar{₀}{\ensuremath{_0}}
\newunicodechar{₁}{\ensuremath{_1}}
\newunicodechar{₂}{\ensuremath{_2}}
\newunicodechar{₃}{\ensuremath{_3}}
\newunicodechar{₄}{\ensuremath{_4}}
\newunicodechar{₅}{\ensuremath{_5}}
\newunicodechar{₆}{\ensuremath{_6}}
\newunicodechar{₇}{\ensuremath{_7}}
\newunicodechar{₈}{\ensuremath{_8}}
\newunicodechar{₉}{\ensuremath{_9}}

% Superscripts (digits not already covered by ², ³)
\newunicodechar{⁰}{\ensuremath{^0}}
\newunicodechar{¹}{\ensuremath{^1}}
\newunicodechar{⁴}{\ensuremath{^4}}
\newunicodechar{⁵}{\ensuremath{^5}}
\newunicodechar{⁶}{\ensuremath{^6}}
\newunicodechar{⁷}{\ensuremath{^7}}
\newunicodechar{⁸}{\ensuremath{^8}}
\newunicodechar{⁹}{\ensuremath{^9}}

% Diagonal arrows
\newunicodechar{↗}{\ensuremath{\nearrow}}
\newunicodechar{↘}{\ensuremath{\searrow}}
\newunicodechar{↖}{\ensuremath{\nwarrow}}
\newunicodechar{↙}{\ensuremath{\swarrow}}

% Box-drawing T-junction (joins existing ─ │ └ ├ family)
\newunicodechar{┬}{+}
\newunicodechar{┴}{+}
\newunicodechar{┼}{+}
\newunicodechar{┐}{+}
\newunicodechar{┘}{+}

% Check / cross (require amssymb in preamble for \checkmark)
\newunicodechar{✓}{\ensuremath{\checkmark}}
\newunicodechar{✗}{\ensuremath{\times}}

% Combining macron: produces X̄, ȳ etc. in prose. The safest fallback
% is the no-op identity so the base char still renders; if accent
% support is wanted, decomposed forms need pre-NFC normalisation in
% the renderer rather than \newunicodechar.
\newunicodechar{̄}{}
```

If `\checkmark` is added, the preamble needs `\usepackage{amssymb}` alongside `\usepackage{newunicodechar}` (line 194).

## Verification path

After landing the fix, the same `rustlab-notebook validate -f pdf notebooks/` run against rustlab_llm should report 24/24 PASS instead of the current 16/24. Build logs are kept at `/tmp/claude-501/rustlab-validate.EcnAbQ/<lesson>/<lesson>.log` for inspection.

## Related notes

- Commit `9a3ba63` ("Notebook LaTeX: declare Unicode chars, bypass inkscape LaTeX bridge, fix \\$ escape", before the 0.3.6 bump) established the declaration block; the fix on `feature/latex-unicode-coverage` is the natural follow-up.
- The trailing-blank-line MD012 issue surfaced in the same validate run is fixed on rustlab branch `feature/renderer-trailing-blanks` (commit `2c7cd99` "markdown renderer: collapse blank-line runs + strip trailing blanks (Bug 2)"). The `MD012: false` suppression in this repo's `.markdownlint-cli2.jsonc` is the right interim until that branch lands and `rustlab-notebook` is republished.
