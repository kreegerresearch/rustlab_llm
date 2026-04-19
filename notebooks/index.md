---
title: "RustLab LLM — A Ground-Up Tour of Language Models"
---

# RustLab LLM

A self-contained tutorial series for building a GPT-style language model
from first principles using [Rustlab](https://github.com/anthropics/rustlab)
as the scripting and visualisation environment.

Each lesson pairs step-by-step mathematical derivation with runnable code
and inline plots. The series builds from raw probability and linear
algebra up to a complete decoder — nothing is a black box.

## Module Roadmap

- **Phase 1 — Foundations** *(Complete)* — Lessons 01–04 cover tokens,
  softmax, cross-entropy, and embeddings.
- **Phase 2 — First Working Model** *(Complete)* — Lessons 05–06 build a
  bigram model and introduce gradient descent on a linear layer.
- **Phase 3 — Attention** *(Complete)* — Lessons 07–09 cover context-aware
  aggregation, scaled dot-product attention, and multi-head attention.
- **Phase 4 — Transformer Components** *(Planned)* — Positional encoding,
  MLP block, LayerNorm and residuals.
- **Phase 5 — Full GPT** *(Planned)* — Assemble the transformer block and
  the complete model.
- **Phase 6–8 — Training, BPE, Generation** *(Planned)*.

## Using the notebooks

Each notebook below can be read straight through — the code blocks run
live against the rustlab runtime and embed their output inline. Every
notebook also has a matching runnable script set under
`lessons/NN-topic/` in the repository, for those who prefer to
experiment at the REPL.

## Lessons
